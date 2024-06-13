from PIL import Image
from typing import Optional, List
from dataclasses import dataclass

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class EncodingOutput(ModelOutput):
    input_ids: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    labels: Optional[torch.LongTensor] = None
    

@dataclass
class PegsOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    loss_gen: Optional[torch.FloatTensor] = None


@dataclass
class PegsGenerationOutput(ModelOutput):
    output_sequence: Optional[torch.LongTensor] = None
    image: Optional[Image.Image] = None


@dataclass
class ConversationalAgentOutput(ModelOutput):
    text_response: Optional[str] = None
    image: Optional[Image.Image] = None