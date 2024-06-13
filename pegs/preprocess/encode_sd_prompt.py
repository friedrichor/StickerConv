import json
import pickle
from tqdm import tqdm

import torch
from diffusers import StableDiffusionPipeline


device = 'cuda:6'
conversation_data_path = "/datas/llm_datasets/kfhcode/StickerConv/instruction_train_1024_double.json"
with open(conversation_data_path, 'r') as file:
    conversation_data = json.load(file)

sd_model = StableDiffusionPipeline.from_pretrained(
    "/datas/huggingface/cuteyukimix_SD1.5/diffusers/"
).to(device)

prompt_embedding_dic = {}
for conv in tqdm(conversation_data):
    prompts = conv["caption"]
    for prompt in prompts:
        if prompt in prompt_embedding_dic.keys():
            continue
        
        with torch.no_grad():
            sd_text_embeddings, _ = sd_model.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                num_images_per_prompt=1,
                device=device
            )
        prompt_embedding_dic[prompt] = sd_text_embeddings.cpu()  # shape: [1, 77, 768]
        
with open("sd_prompt_embedding.pkl", "wb") as file:
    pickle.dump(prompt_embedding_dic, file)
