import logging
import contextlib
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
from torch.nn.utils import rnn
from transformers import (LlamaTokenizer, LlamaForCausalLM, Blip2ForConditionalGeneration,
                          BitsAndBytesConfig, StoppingCriteriaList)
from transformers.modeling_outputs import CausalLMOutput
from peft import LoraConfig, get_peft_model, TaskType

from src.model.config import PegsGenConfig
from src.model.projection_layers import InputProjector, OutputProjector
from src.model.utils import StoppingCriteriaSub, disabled_train, convert_weights_to_fp16
from src.model.outputs import EncodingOutput, PegsOutput, PegsGenerationOutput
from src.common import registry


@registry.register_model("pegs_gen")
class PegsGen(nn.Module):
    def __init__(self, config: PegsGenConfig):
        super().__init__()
        self.config = config
        
        # llm
        logging.info("Loading LLM...")
        self.text_config = config.text_config
        self.tokenizer = LlamaTokenizer.from_pretrained(config.text_config._name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if config.low_resource:
            self.text_model = LlamaForCausalLM.from_pretrained(
                config.text_config._name_or_path,
                torch_dtype=torch.float16,
                device_map={'': config.device_8bit},
                load_in_8bit=True
            )
        else:
            self.text_model = LlamaForCausalLM.from_pretrained(
                config.text_config._name_or_path,
                torch_dtype=torch.float16
            )
        
        self.image_start_token = config.image_start_token
        self.image_end_token   = config.image_end_token
        self.image_placeholder_token = config.image_placeholder_token
        if config.enable_generation:
            self._add_image_tokens(config.num_image_tokens)
        else:
            self._add_image_tokens(0)
        
        if config.lora_config is not None and config.lora_config.lora_enable:
            logging.info("Use LoRA")
            lora_config = LoraConfig(
                r=config.lora_config.lora_r,
                lora_alpha=config.lora_config.lora_alpha,
                target_modules=config.lora_config.lora_target_modules,
                lora_dropout=config.lora_config.lora_dropout,
                bias=config.lora_config.lora_bias,
                modules_to_save=config.lora_config.modules_to_save,
                task_type=TaskType.CAUSAL_LM,
            )
            self.text_model = get_peft_model(self.text_model, lora_config)
            self.text_model.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
            self.text_model.base_model.model.lm_head.original_module.weight.requires_grad = False
        else:
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False
        logging.info("LLM has been loaded.")
        
        self.max_text_length = config.max_text_length
        self.max_length = config.max_length
        self.prefix_prompt = config.prefix_prompt
        
        # image perception
        if config.enable_perception:
            logging.info("Loading Vision Encoder and Q-Former...")
            self.initialize_vision_modules()
            self.inputProjector = InputProjector(self.qformer_config.hidden_size, self.text_config.hidden_size)
            
        # image generation
        if config.enable_generation:
            self.outputProjector = OutputProjector(self.text_config.hidden_size, config.num_clip_tokens, config.prompt_embeddings_dim)

            self.num_image_tokens = config.num_image_tokens
            self._add_image_tokens(self.num_image_tokens)
            with torch.no_grad():
                self.input_embeds_grad_mask = torch.ones_like(self.text_model.get_input_embeddings().weight.data)
                self.output_embeds_grad_mask = torch.ones_like(self.text_model.get_output_embeddings().weight.data)
                self.input_embeds_grad_mask[:-self.num_image_tokens] = 0
                self.output_embeds_grad_mask[:-self.num_image_tokens] = 0     
            
    def initialize_vision_modules(self):
        config: PegsGenConfig = self.config
        
        blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            config.vision_config._name_or_path
        )
        self.vision_encoder = blip2_model.vision_model
        self.qformer = blip2_model.qformer
        self.query_tokens = blip2_model.query_tokens  # [1, 32, 768]
        del blip2_model
        
        if config.vision_precision == "fp16":
            convert_weights_to_fp16(self.vision_encoder)
        
        if config.freeze_vision_encoder:
            for _, param in self.vision_encoder.named_parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train
            logging.info("Freeze Vision Encoder.")
        logging.info("Vision Encoder has been loaded.")
        
        self.qformer_config = config.vision_config.qformer_config
        if config.freeze_qformer:
            for _, param in self.qformer.named_parameters():
                param.requires_grad = False
            self.qformer.eval()
            self.qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("Freeze Q-Former.")
        logging.info("Q-Former has been loaded.")

    def _add_image_tokens(self, num_image_tokens: int = 0):
        special_tokens = [self.image_start_token, self.image_end_token]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        logging.info(f"\nadd special tokens: {special_tokens}")
        logging.info(f"\nspecial_tokens_ids: {self.tokenizer.convert_tokens_to_ids(special_tokens)}")
        
        if self.config.enable_generation:
            self.image_tokens = [f"[IMG{i+1}]" for i in range(num_image_tokens)]
            logging.info(f"\nadd image tokens: {self.image_tokens}")
            for image_token in self.image_tokens:
                self.tokenizer.add_tokens([image_token])
            self.image_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.image_tokens)
            logging.info(f"\nimage tokens ids: {self.image_tokens_ids}")
        
        # resize token embeddings
        if len(self.tokenizer) > self.tokenizer.vocab_size:
            self.text_model.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.save_pretrained("tokenizer")
        
        # embeddings
        with torch.no_grad():
            device = self.text_model.device
            self.image_start_token_id = self.tokenizer(self.image_start_token, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            self.image_end_token_id = self.tokenizer(self.image_end_token, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            
            self.image_start_embeds = self.get_input_embeddings()(self.image_start_token_id)
            self.image_end_embeds = self.get_input_embeddings()(self.image_end_token_id)
            
            if self.config.enable_generation:
                self.image_tokens_input_ids = torch.tensor(self.image_tokens_ids).unsqueeze(0).to(device)  # [1, num_image_tokens]
                self.image_tokens_attention_mask = torch.ones(self.image_tokens_input_ids.size(), dtype=torch.long, device=device)  # [1, num_image_tokens]
                self.image_tokens_labels = torch.tensor(self.image_tokens_ids).unsqueeze(0).to(device)  # [1, num_image_tokens]

    @property
    def image_tokens_embeds(self):
        return self.get_input_embeddings()(self.image_tokens_input_ids)
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.text_model.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def reset_embeddings(self):
        with torch.no_grad():
            if self.config.lora_config.lora_enable:
                for name, param in self.text_model.named_parameters():
                    if param.grad is None:
                        continue
                    if "embed_tokens" in name:
                        param.grad = param.grad * self.input_embeds_grad_mask.to(param.device)
                    elif "lm_head" in name:
                        param.grad = param.grad * self.output_embeds_grad_mask.to(param.device)
            else:
                self.text_model.get_input_embeddings().weight.grad = self.text_model.get_input_embeddings().weight.grad * self.input_embeds_grad_mask.to(self.llm_model.device)
                if self.config.enable_generation:
                    self.text_model.get_output_embeddings().weight.grad = self.text_model.get_output_embeddings().weight.grad * self.output_embeds_grad_mask.to(self.llm_model.device)
    
    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()
        
    def encode_image(self, pixel_values: torch.FloatTensor):
        device = self.vision_encoder.device
        
        with self.maybe_autocast():  # Required, otherwise an error "RuntimeError: expected scalar type Half but found Float" will be reported.
            vision_outputs = self.vision_encoder(pixel_values.to(device))  # pixel_values: [bs, 3, 224, 224]
        
            image_embeds = vision_outputs.last_hidden_state.to(device)  # [bs, 257(=1+16*16), 1408]
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)  # [bs, 257]
        
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask
            )  # last_hidden_state, pooler_output, hidden_states, past_key_values, attentions, cross_attentions
            query_output = query_outputs.last_hidden_state  # [bs, 32, 768]  # query tokens num: 32

            image_content_embeds = self.inputProjector(query_output)  # [bs, 32, 4096]
            image_start_embeds = self.image_start_embeds.expand(image_embeds.shape[0], -1, -1).to(device)  # [bs, 1, 4096]
            image_end_embeds   = self.image_end_embeds.expand(image_embeds.shape[0], -1, -1).to(device)    # [bs, 1, 4096]

            image_embeds = torch.cat([image_start_embeds, image_content_embeds, image_end_embeds], dim=1)  # [bs, 34(1+32+1), 4096]
            attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)  # [bs, 34(1+32+1)]
            labels = attention_mask * (-100)
            
        return EncodingOutput(
            inputs_embeds=image_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def encode_text(self, text: Union[str, List[str]]):
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_text_length,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.text_model.device)
        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
        inputs_embeds = self.get_input_embeddings()(input_ids)  # [bs, seq len, 4096]

        return EncodingOutput(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
    
    def encode_prefix_prompt(self, batch_size, prefix_prompt: Optional[str] = None):
        device = self.text_model.device
        # bos
        bos_input_ids = torch.ones([batch_size, 1], dtype=torch.int64, device=device) * self.tokenizer.bos_token_id
        bos_embeds = self.get_input_embeddings()(bos_input_ids)  # [bs, 1, 4096]
        bos_attention_mask = torch.ones([batch_size, 1], dtype=torch.int64, device=device)  # [bs, 1]
        bos_labels = bos_attention_mask * (-100)
        
        if prefix_prompt is not None:
            encoded_prefix_prompt = self.encode_text([prefix_prompt] * batch_size)
            
            input_ids      = torch.cat([bos_input_ids, encoded_prefix_prompt.input_ids], dim=1)
            inputs_embeds  = torch.cat([bos_embeds, encoded_prefix_prompt.inputs_embeds], dim=1)
            attention_mask = torch.cat([bos_attention_mask, encoded_prefix_prompt.attention_mask], dim=1)
            labels         = torch.cat([bos_labels, encoded_prefix_prompt.attention_mask * (-100)], dim=1)
            
            return EncodingOutput(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            return EncodingOutput(
                input_ids=bos_input_ids, 
                inputs_embeds=bos_embeds,
                attention_mask=bos_attention_mask,
                labels=bos_labels
            )
    
    def build_one_batch_perception(self, pixel_values: torch.FloatTensor, captions: List[str], **kwargs):
        encoded_images   = self.encode_image(pixel_values)
        encoded_captions = self.encode_text(captions)
        encoded_bos      = self.encode_prefix_prompt(encoded_captions.inputs_embeds.shape[0], None)
        
        inputs_embeds = torch.cat([encoded_bos.inputs_embeds, encoded_images.inputs_embeds, encoded_captions.inputs_embeds], dim=1)
        attention_mask = torch.cat([encoded_bos.attention_mask,encoded_images.attention_mask,encoded_captions.attention_mask], dim=1)
        labels = torch.cat([
            encoded_bos.labels,
            encoded_images.labels,
            encoded_captions.input_ids.masked_fill(encoded_captions.input_ids == self.tokenizer.pad_token_id, -100)
        ], dim=1)
        assert inputs_embeds.shape[:-1] == attention_mask.shape == labels.shape
        
        return EncodingOutput(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

    def build_one_batch_generation(self, captions: List[str], **kwargs):
        input_ids_list, inputs_embeds_list, attention_mask_list, labels_list = [], [], [], []
        encoded_bos = self.encode_prefix_prompt(1, None)
        for caption in captions:
            encoded_caption = self.encode_text(caption, max_length=64)
            
            """total_seq_len = 1 + seq_len + num_image_tokens"""
            input_ids = torch.cat([encoded_bos.input_ids, encoded_caption.input_ids, self.image_tokens_input_ids], dim=1)  # [1, total_seq_len])
            inputs_embeds = torch.cat([encoded_bos.inputs_embeds, encoded_caption.inputs_embeds, self.image_tokens_embeds], dim=1)  # [1, total_seq_len, 4096])
            attention_mask = torch.cat([encoded_bos.attention_mask, encoded_caption.attention_mask, self.image_tokens_attention_mask], dim=1)  # [1, total_seq_len]
            labels = torch.cat([
                encoded_bos.labels,
                encoded_caption.attention_mask * (-100),
                self.image_tokens_labels
            ], dim=1)  # [1, 1+seq_len+num_image_tokens]
            
            input_ids_list.append(input_ids.squeeze(0))
            inputs_embeds_list.append(inputs_embeds.squeeze(0))
            attention_mask_list.append(attention_mask.squeeze(0))
            labels_list.append(labels.squeeze(0))
        
        input_ids = rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)  # [bs, total_seq_len]
        inputs_embeds = rnn.pad_sequence(inputs_embeds_list, batch_first=True)  # [bs, total_seq_len, 4096]
        attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)  # [bs, total_seq_len]
        labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)  # [bs, total_seq_len]
        assert input_ids.shape == inputs_embeds.shape[:-1] == attention_mask.shape == labels.shape
        
        return EncodingOutput(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def build_one_instance_perception_and_generation(self, pixel_values: torch.FloatTensor, text: str):
        text_list = text.split(self.image_placeholder_token)
        
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        for i, one_text in enumerate(text_list):
            if one_text != "":  # text
                one_input_ids, one_text_embeds, one_text_attention_mask = self.encode_text(one_text)
                
                input_ids_list.append(one_input_ids)  # one_input_ids: [1, seq len]
                input_embeds_list.append(one_text_embeds)  # one_text_embeds: [1, seq len, 4096]
                input_attention_mask_list.append(one_text_attention_mask)  # one_text_attention_mask: [1, seq len]
                labels_list.append(one_input_ids.masked_fill(one_input_ids == self.tokenizer.pad_token_id, -100))
            if i != len(text_list) - 1:  # image
                if "Human" == one_text.split("### ")[-1][:5]:  # image for perception
                    one_image_embeds, one_image_attention_mask, one_image_labels = self.encode_image(pixel_values[i].unsqueeze(0))
                    one_input_ids = torch.zeros(one_image_labels.size(), dtype=torch.long, device=self.llm_model.device)  # Just as a placeholder input_ids. Don't affect training
                    
                    input_ids_list.append(one_input_ids)  # one_input_ids: [1, 34(=1+32+1)]
                    input_embeds_list.append(one_image_embeds)  # one_image_embeds: [1, 34(=1+32+1), 4096]
                    input_attention_mask_list.append(one_image_attention_mask)  # one_image_attention_mask: [1, 34(=1+32+1)]
                    labels_list.append(one_image_labels)  # one_image_labels: [1, 34(=1+32+1)]
                elif "Assistant" == one_text.split("### ")[-1][:9]:   # image for generation
                    input_ids_list.append(self.image_tokens_input_ids)  # image_tokens_input_ids: [1, 32]
                    input_embeds_list.append(self.image_tokens_embeds)  # image_tokens_embeds: [1, 32, 4096]
                    input_attention_mask_list.append(self.image_tokens_attention_mask)  # image_tokens_attention_mask: [1, 32]
                    labels_list.append(self.image_tokens_labels)  # image_tokens_labels: [1, 32]
                else:
                    logging.warning(f"Training data is irregular!\n{one_text.split('###')[-1]}")

        bos_input_ids, bos_embeds, bos_attention_mask, bos_labels = self.get_encoded_bos(1)
        input_ids = torch.cat([bos_input_ids] + input_ids_list, dim=1)  # [1, seq len]
        input_embeds = torch.cat([bos_embeds] + input_embeds_list, dim=1)  #[1, seq len, 4096]
        input_attention_mask = torch.cat([bos_attention_mask] + input_attention_mask_list, dim=1)  # [1, seq len]
        labels = torch.cat([bos_labels] + labels_list, dim=1)  # [1, seq len]
        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape
        
        return input_ids, input_embeds, input_attention_mask, labels
    
    def build_one_batch_perception_and_generation(self, pixel_values: torch.FloatTensor, text: List[str]):
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        for one_pixel_values, one_text in zip(pixel_values, text):
            one_input_ids, one_input_embeds, one_input_attention_mask, one_labels = self.build_one_instance_perception_and_generation(one_pixel_values, one_text)

            input_ids_list.append(one_input_ids.squeeze(0))
            input_embeds_list.append(one_input_embeds.squeeze(0))
            input_attention_mask_list.append(one_input_attention_mask.squeeze(0))
            labels_list.append(one_labels.squeeze(0))
        
        input_ids = rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)[:, :self.max_text_length]  # [b_s, min(seq len, max_text_length)]
        input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)[:, :self.max_text_length, :]  # [b_s, min(seq len, max_text_length), 4096]
        input_attention_mask = rnn.pad_sequence(input_attention_mask_list, batch_first=True, padding_value=0)[:, :self.max_text_length]  # [b_s, min(seq len, max_text_length)]
        labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)[:, :self.max_text_length]  # [b_s, min(seq len, max_text_length)]
        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape

        return input_ids, input_embeds, input_attention_mask, labels

    def forward(
        self,
        caption: Union[str, List[str]] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        conversation: Optional[List[str]] = None,
        prompt_embeddings_dic: Optional[Dict[str, torch.FloatTensor]] = None
    ) -> CausalLMOutput:
        if self.config.enable_perception and not self.config.enable_generation:
            build_batch_function = self.build_one_batch_perception
        elif not self.config.enable_perception and self.config.enable_generation:
            build_batch_function = self.build_one_batch_generation
        elif self.config.enable_perception and self.config.enable_generation:
            build_batch_function = self.build_one_batch_perception_and_generation
        
        batch = build_batch_function(pixel_values=pixel_values, captions=caption, conversation=conversation)

        outputs = self.text_model(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            return_dict=True,
            output_hidden_states=True,
        )  # 'loss', 'logits', 'past_key_values', 'hidden_states'
        lm_loss = outputs.loss
        
        if self.config.enable_generation:
            mse_loss = 0
            mse_loss_function = nn.MSELoss()
            if not self.config.enable_perception:  # pre-train generation-only
                image_start_position = (batch.labels == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 1].tolist()
                image_end_position = (batch.labels == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 1].tolist()
                
                hidden_states_list = []
                for b, (start, end) in enumerate(zip(image_start_position, image_end_position)):
                    hidden_states_list.append(outputs.hidden_states[-1][b, start: end + 1, :])
                hidden_states = torch.stack(hidden_states_list, dim=0)  # [bs, num_image_tokens, 4096]
                projected_hidden_states = self.outputProjector(hidden_states)  # [bs, 77, 768]

                sd_text_embeddings_list = []
                for c in caption:
                    sd_text_embeddings_list.append(prompt_embeddings_dic[c])
                sd_text_embeddings = torch.stack(sd_text_embeddings_list, dim=0)  # [bs, 77, 768]
                
                mse_loss = mse_loss_function(projected_hidden_states, sd_text_embeddings)
                mse_loss = mse_loss.mean()
                
                return PegsOutput(loss=lm_loss + mse_loss, loss_lm=lm_loss, loss_gen=mse_loss)
                
            else:  # joint learning
                count_system_have_sticker = 0
                for b, (per_labels, per_caption) in enumerate(zip(batch.labels, caption)):
                    image_start_position = (per_labels == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 0].tolist()
                    image_end_position = (per_labels == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 0].tolist()
                    num_image = min(len(image_start_position), len(image_end_position))
                    
                    if num_image > 0:
                        count_system_have_sticker += 1
                        image_start_position, image_end_position = image_start_position[:num_image], image_end_position[:num_image]
                        
                        hidden_states_list = []
                        for (start, end) in zip(image_start_position, image_end_position):
                            hidden_states_list.append(outputs.hidden_states[-1][b, start: end + 1, :])
                        hidden_states = torch.stack(hidden_states_list, dim=0)  # [num_image, 32, 4096]
                        projected_hidden_states = self.outputProjector(hidden_states)  # [num_image, 77, 768]

                        sd_text_embeddings_list = []
                        for c in per_caption:
                            sd_text_embeddings_list.append(prompt_embeddings_dic[c])
                        sd_text_embeddings = torch.stack(sd_text_embeddings_list, dim=0)  # [num_image, 77, 768]
                    
                        mse_loss += mse_loss_function(projected_hidden_states, sd_text_embeddings)
                    
                if count_system_have_sticker > 0:
                    mse_loss = mse_loss / count_system_have_sticker
                    
                return PegsOutput(loss=lm_loss + mse_loss, loss_lm=lm_loss, loss_gen=mse_loss)
        else:
            return PegsOutput(loss=lm_loss, loss_lm=lm_loss, loss_gen=0)
 
    @torch.no_grad()
    def generate(
        self, 
        pixel_values: Optional[torch.FloatTensor] = None,
        text: Optional[str] = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        min_length: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negativate_prompt: Optional[str] = None,
    ):
        generated_image = None
        # generation setting
        stop_words_ids = [
            torch.tensor([835]).to(self.llm_model.device),          # '###' can be encoded in two different ways.
            torch.tensor([2277, 29937]).to(self.llm_model.device),  # {"###": 835, "##": 2277, "#": 29937}
            torch.tensor([32033]).to(self.llm_model.device)         # [IMG32]
        ]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        if self.enable_perception and not self.enable_generation:
            _, input_embeds, input_attention_mask, _ = self.build_one_batch_perception(pixel_values, text)  # [1, seq_len, 4096], [1, seq_len]
            input_embeds = input_embeds[:, -self.max_text_length:, :]  # [1, min(seq_len, max_text_length), 4096]
            input_attention_mask = input_attention_mask[:, -self.max_text_length:]  # [1, min(seq_len, max_text_length)]
            
            outputs = self.llm_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_attention_mask,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=6,
                stopping_criteria=stopping_criteria,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )  # 'sequences', 'hidden_states', 'past_key_values'
            output_sequence = outputs.sequences[0][1:]
            
        elif not self.enable_perception and self.enable_generation:
            _, input_embeds, input_attention_mask, labels = self.build_one_batch_generation([text])

            outputs = self.llm_model(
                inputs_embeds=input_embeds,
                attention_mask=input_attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            print(f"outputs.hidden_states[-1].shape = {outputs.hidden_states[-1].shape}")
            hidden_states_list = []
            image_start_position = (labels == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 1].tolist()  # len = b_s
            image_end_position = (labels == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 1].tolist()  # len = b_s
            # print(outputs.hidden_states[-1].shape)  # torch.Size([b_s, seq len, 4096])
            print(f"image_start_position = {image_start_position}")
            print(f"image_end_position = {image_end_position}")
            for b, (start, end) in enumerate(zip(image_start_position, image_end_position)):
                hidden_states_list.append(outputs.hidden_states[-1][b, start: end + 1, :])
            hidden_states = torch.stack(hidden_states_list, dim=0)  # torch.Size([b_s, 32, 4096])
            print(f"hidden_states.shape = {hidden_states.shape}")
            projected_hidden_states = self.outputProjector(hidden_states)  # torch.Size([b_s, 77, 768])
            print(f"projected_hidden_states.shape = {projected_hidden_states.shape}")
            
            outputs = self.stable_diffusion(
                prompt_embeds=projected_hidden_states,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negativate_prompt,
            )
            
            output_sequence = None
            generated_image = outputs.images[0]
        else:
            _, input_embeds, input_attention_mask, _ = self.build_one_instance_for_demo(pixel_values, text)
            outputs = self.llm_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_attention_mask,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=6,
                stopping_criteria=stopping_criteria,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )  # 'sequences', 'hidden_states', 'past_key_values'
            output_sequence = outputs.sequences[0][1:]
            output_embeddings = []
            for _hidden_states in outputs.hidden_states[1:]:
                output_embeddings.append(_hidden_states[-1])
            output_hidden_states = torch.cat(output_embeddings, dim=1)
            
            hidden_states_list = []
            if self.image_tokens_ids[0] in output_sequence:
                image_start_position = (output_sequence == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 0].tolist()[0]  # len = b_s
                image_end_position = (output_sequence == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 0].tolist()[0]  # len = b_s
                
                if image_end_position + 1 - image_start_position == self.num_image_tokens:
                    gen_hidden_states = output_hidden_states[:, image_start_position: image_end_position + 1, :]
                    projected_hidden_states = self.outputProjector(gen_hidden_states)  # torch.Size([b_s, 77, 768])

                    logging.info(f"negativate_prompt = \n{negativate_prompt}")
                    random_seed = torch.seed()
                    generator = torch.Generator(device="cuda").manual_seed(random_seed)
                    sd_outputs = self.stable_diffusion(
                        prompt_embeds=projected_hidden_states,
                        negativate_prompt=negativate_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )
                    logging.info(f"is_NSFW: {sd_outputs.nsfw_content_detected[0]}")
                    if sd_outputs.nsfw_content_detected[0] == False:
                        generated_image = sd_outputs.images[0]
            
        
        return PegsGenerationOutput(
            output_sequence=output_sequence,
            image=generated_image
        )
        
        
        
        
        