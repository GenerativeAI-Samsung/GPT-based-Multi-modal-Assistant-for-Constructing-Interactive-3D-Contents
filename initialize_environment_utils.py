from PIL import Image

import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def forward(self, text, image_paths):
        images = [Image.open(item) for item in image_paths]
        
        inputs = self.processor(text=text, images=images)
        outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image  
        probs = logits_per_image.softmax(dim=0)

        return probs 

# class DiffusionModel(nn.Module):