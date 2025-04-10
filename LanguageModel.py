import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import TextStreamer
from peft import PeftModel

import random
import asyncio
import g4f

from PIL import Image

import sys
sys.path.append('/content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents')

class VisionLangugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", 
                                               torch_dtype=torch.float16, 
                                               trust_remote_code=True).to("cuda")
        self.processor = AutoProcessor.from_pretrained("visheratin/MC-LLaVA-3b", 
                                                       trust_remote_code=True)
    
    def process(self, image_path=None, query=None):
        raw_image = Image.open(image_path)

        prompt = f"""<|im_start|>user
                    <image>
                    {query}<|im_end|>
                    <|im_start|>assistant
                """
        
        with torch.inference_mode():
            inputs = self.processor(prompt, 
                                    [raw_image], 
                                    self.model, 
                                    max_crops=100, 
                                    num_tokens=728)
            
        streamer = TextStreamer(self.processor.tokenizer)
        with torch.inference_mode():
            output = self.model.generate(**inputs, 
                                         max_new_tokens=200, 
                                         do_sample=True, 
                                         use_cache=False, 
                                         top_p=0.9, 
                                         temperature=1.2, 
                                         eos_token_id=self.processor.tokenizer.eos_token_id, 
                                         streamer=streamer)
        return self.processor.tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", "")

class UserInteractModel(nn.Module):
    def __init__(self, MODEL_ID):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                                       model_max_length=1536)
        self.tokenizer.pad_token = self.tokenizer.pad_token
        
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        
        self.smart_tokenizer_and_embedding()
    
    def add_special_tokens(self):
        default_pad_token = "[PAD]"
        default_eos_token = "</s>"
        default_bos_token = "<s>"
        default_unk_token = "<unk>"

        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = default_pad_token
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = default_eos_token
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = default_bos_token
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = default_unk_token
        
        if special_tokens_dict:
            num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
            return num_added_tokens
        return 0 
    
    def smart_tokenizer_and_embedding(self):
        num_new_tokens = self.add_special_tokens()

        if num_new_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

            input_embeddings = self.model.get_input_embeddings().weight.data
            output_embeddings = self.model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


    def generate(self, batch):
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, max_length=1024, output_hidden_states=True, return_dict_in_generate=True)

        respone = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        return respone

class ScriptToScene(nn.Module):
    def __init__(self):
        super().__init__()
        self.TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"
        self.STEP1_ID = "/content/drive/MyDrive/Text_to_scene/Step1_TextToScene"
        self.STEP2_ID = "/content/drive/MyDrive/Text_to_scene/Step2_TextToScene"
        self.STEP3_ID = "/content/drive/MyDrive/Text_to_scene/Step3_TextToScene"
        self.STEP4_ID = "/content/drive/MyDrive/Text_to_scene/Step4_TextToScene"
        self.STEP5_ID = "/content/drive/MyDrive/Text_to_scene/Step5_TextToScene"

        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_ID, use_fast=True)
        self.step1_textToScene = AutoModelForCausalLM.from_pretrained(self.STEP1_ID, device_map="auto", trust_remote_code=True)
        self.step2_textToScene = AutoModelForCausalLM.from_pretrained(self.STEP2_ID, device_map="auto", trust_remote_code=True)
        self.step3_textToScene = AutoModelForCausalLM.from_pretrained(self.STEP3_ID, device_map="auto", trust_remote_code=True)
        self.step4_textToScene = AutoModelForCausalLM.from_pretrained(self.STEP4_ID, device_map="auto", trust_remote_code=True)
        self.step5_textToScene = AutoModelForCausalLM.from_pretrained(self.STEP5_ID, device_map="auto", trust_remote_code=True)

    def generate(self, model, prompt):
        tok_prompt = self.tokenizer(prompt, return_tensors='pt')
        tok_prompt = tok_prompt.to('cuda:0')

        output_ids = model.generate(**tok_prompt, max_new_tokens=1024)

        output_txt = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_txt = output_txt[len(prompt):].strip()
        return generated_txt        

    def step1(self, request):
        prompt = request + ' | '
        generated_txt = self.generate(prompt, self.step1_textToScene)
        return generated_txt

    def step2(self, request, object_list):
        prompt = request + ' | ' + object_list + ' | '
        generated_txt = self.generate(prompt, self.step2_textToScene)
        return generated_txt

    def step3(self, request, object_list, init_pos):
        prompt = request + ' | ' + object_list + ' | ' + init_pos + ' | '
        generated_txt = self.generate(prompt, self.step3_textToScene)
        return generated_txt

    def step4(self, request):
        prompt = request + ' | ' 
        generated_txt = self.generate(prompt, self.step4_textToScene)
        return generated_txt
    
    def step5(self, request, object_evironment_list):
        prompt = request + ' | ' + object_evironment_list + ' | '
        generated_txt = self.generate(prompt, self.step5_textToScene)
        return generated_txt

class ModifyPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"
        self.CLASSIFY_ID = "/content/drive/MyDrive/Text_to_scene/ClassifyStep"
        self.STEP1_ID = "/content/drive/MyDrive/Text_to_scene/Step1_ModifyPart"
        self.STEP2_ID = "/content/drive/MyDrive/Text_to_scene/Step2_ModifyPart"
        self.STEP3_ID = "/content/drive/MyDrive/Text_to_scene/Step3_ModifyPart"

        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_ID, use_fast=True)
        self.classify = AutoModelForCausalLM.from_pretrained(self.CLASSIFY_ID, device_map="auto", trust_remote_code=True)
        self.step1_modifyPart = AutoModelForCausalLM.from_pretrained(self.STEP1_ID, device_map="auto", trust_remote_code=True)
        self.step2_modifyPart = AutoModelForCausalLM.from_pretrained(self.STEP2_ID, device_map="auto", trust_remote_code=True)
        self.step3_modifyPart = AutoModelForCausalLM.from_pretrained(self.STEP3_ID, device_map="auto", trust_remote_code=True)

    def generate(self, model, prompt):
        tok_prompt = self.tokenizer(prompt, return_tensors='pt')
        tok_prompt = tok_prompt.to('cuda:0')

        output_ids = model.generate(**tok_prompt, max_new_tokens=1024)

        output_txt = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_txt = output_txt[len(prompt):].strip()
        return generated_txt        

    def step1_modify(self, ori_req, ori_object_list, modify_res):
        prompt = ori_req + ' | ' + ori_object_list + ' | ' + modify_res + ' | '
        generated_txt = self.generate(prompt, self.step1_modifyPart)
        return generated_txt

    def step2_modify(self, ori_req, modified_object_list, ori_init_pos, modify_res):
        prompt = ori_req + ' | ' + modified_object_list + ' | ' + ori_init_pos + ' | ' + modify_res + ' | '
        generated_txt = self.generate(prompt, self.step2_modifyPart)
        return generated_txt

    def step3_modify(self, ori_req, ori_movs, modify_res):
        prompt = ori_req + ' | ' + ori_movs + ' | '  + modify_res + ' | '
        generated_txt = self.generate(prompt, self.step2_modifyPart)
        return generated_txt

async def deepseek_generate(prompt):
    async def process_api_request(request, index):
        tries = 0
        while (True):
            try:
                await asyncio.sleep(random.randint(5, 10))
                print(f"Started API request of index: {index}.")
                response = await g4f.ChatCompletion.create_async(
                    model="deepseek-v3",
                    messages=[{"role": "user", "content": request}],
                )
                print(response)
                print(f"Completed API request of index: {index}")
                return response
            except Exception as e:
                print(f"Request of index {index} - try: {tries} - Error: {str(e)}")
                if (tries == 3):
                  return [None]
                tries += 1
                await asyncio.sleep(5)
    tasks = []
    for index, request in enumerate(prompt):
        tasks.append(process_api_request(request, index))
    return await asyncio.gather(*tasks, return_exceptions=True)
