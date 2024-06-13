import json
from typing import Optional, List
from omegaconf import DictConfig
from dataclasses import dataclass, field, asdict, is_dataclass
from transformers import LlamaConfig, Blip2Config


@dataclass
class CustomLoraConfig:
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    lora_target_modules: Optional[List] = field(default=None)
    modules_to_save: Optional[List] = field(default=None)
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules=["q_proj","v_proj"]
            
        if self.modules_to_save is None:
            self.modules_to_save=['lm_head','embed_tokens']


class BasePegsConfig:
    model_type = "base_pegs"
    def __init__(
        self,
        text_model: str,
        lora_config: Optional[DictConfig] = None,
        max_text_length: int = 1024,
        max_length: int = 1024,
        prefix_prompt: Optional[str] = None,
        low_resource: bool = False,  # use 8 bit
        device_8bit: int = 0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        vision_modules: Optional[str] = None,  # clip and qformer
        freeze_vision_encoder: bool = True,
        freeze_qformer: bool = True,
        vision_precision: str = "fp16",
        image_start_token: str = "<Img>",
        image_end_token: str = "</Img>",
        image_placeholder_token: str = "<IMG>",
        **kwargs
    ) -> None:
        self.text_config = LlamaConfig.from_pretrained(text_model)
        self.text_config._name_or_path = text_model
        
        self.lora_config = CustomLoraConfig(lora_config)
        
        self.max_text_length = max_text_length
        self.max_length = max_length
        self.prefix_prompt = prefix_prompt
        
        self.low_resource = low_resource
        self.device_8bit = device_8bit
        
        if vision_modules is not None:
            self.enable_perception = True
            
            self.vision_config = Blip2Config.from_pretrained(vision_modules)
            self.vision_config._name_or_path = vision_modules
            
            self.freeze_vision_encoder = freeze_vision_encoder
            self.freeze_qformer        = freeze_qformer
            self.vision_precision      = vision_precision
        else:
            self.enable_perception = False
        
        self.image_start_token = image_start_token
        self.image_end_token   = image_end_token
        self.image_placeholder_token = image_placeholder_token
        
    def to_dict(self):
        return self.__dict__
            
       
class PegsGenConfig(BasePegsConfig):
    model_type = "pegs_gen"
    def __init__(
        self, 
        text_model: str,
        lora_config: Optional[DictConfig] = None,
        max_text_length: int = 1024,
        max_length: int = 1024,
        prefix_prompt: Optional[str] = None,
        low_resource: bool = False,
        device_8bit: int = 0,
        vision_modules: Optional[str] = None,  # clip and qformer
        freeze_vision_encoder: bool = True,
        freeze_qformer: bool = True,
        vision_precision: str = "fp16",
        image_start_token: str = "<Img>",
        image_end_token: str = "</Img>",
        image_placeholder_token: str = "<IMG>",
        image_decoder: Optional[str] = None,   # stable diffusion
        freeze_image_decoder: bool = True,
        num_clip_tokens: int = 77,
        prompt_embeddings_dim: int = 768,  # SD1.5 -> 768, SD>=2.0 -> 1024
        num_image_tokens: int = 32,
        **kwargs
    ) -> None:
        super().__init__(text_model, lora_config, max_text_length, max_length, prefix_prompt, low_resource, device_8bit, vision_modules, freeze_vision_encoder, freeze_qformer, vision_precision, image_start_token, image_end_token, image_placeholder_token, **kwargs)
        
        if image_decoder is not None:
            self.enable_generation = True
            
            self.image_decoder = image_decoder
            
            self.freeze_image_decoder = freeze_image_decoder
            self.num_clip_tokens = num_clip_tokens
            self.prompt_embeddings_dim = prompt_embeddings_dim
            self.num_image_tokens = num_image_tokens
        else:
            self.enable_generation = False


class ModelConfig:
    def __init__(self, config: DictConfig):
        model_config_mapping = {
            PegsGenConfig.model_type: PegsGenConfig
        }
        model_type = config.get('model_type', None)
        if model_type is None:
            raise ModuleNotFoundError("Can't find the parameter 'model_type',' please check the yaml file")
        self.config = model_config_mapping[model_type](**config)
    
    def get_model_config(self):
        return self.config


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if is_dataclass(obj):
            return asdict(obj)
        return json.JSONEncoder.default(self, obj)
    
    
if __name__ == '__main__':
    dict_config = DictConfig({
        'model_type': 'pegs_gen',
        'text_model': '/datas/huggingface/vicuna-7b-v1.5-16k'
    })
    print(dict_config)
    model_config = ModelConfig(dict_config).config
    print(json.dumps(model_config, indent=4, cls=CustomJSONEncoder))
