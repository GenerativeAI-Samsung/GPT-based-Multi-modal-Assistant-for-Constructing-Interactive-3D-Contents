import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import TextStreamer
from peft import PeftModel

from PIL import Image

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

class ScenePlanningModel(nn.Module):
    def __init__(self, 
                 MODEL_ID,
                 TOKENIZER_ID, 
                 adapter_layer1=None, 
                 adapter_layer2=None, 
                 adapter_layer3=None,
                 adapter_layer4=None,
                 adapter_layer5=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID,
                                                       model_max_length=4096,
                                                       use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", trust_remote_code=True
        )
        
        if (adapter_layer1 != None):
            self.step1_layer = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer1,
                                                        is_trainable=False)

        if (adapter_layer2 != None):
            self.step2_layer = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer2,
                                                        is_trainable=False)

        if (adapter_layer3 != None):
            self.step3_layer = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer3,
                                                        is_trainable=False)

        if (adapter_layer4 != None):
            self.step4_layer = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer4,
                                                        is_trainable=False)

        if (adapter_layer5 != None):
            self.step5_layer = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer5,
                                                        is_trainable=False)

    def step1_generate(self, request):
        prompt = request

        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step1_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()

        return generated_text

    def step2_generate(self, request, objects_list):
        prompt = request + ' | ' + objects_list + ' | '

        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step2_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()

        return generated_text

    def step3_generate(self, request, objects_list, init_pos_ori):
        prompt = request + ' | ' + objects_list + ' | ' + init_pos_ori + ' | '

        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step3_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()

        return generated_text
    
    def step4_generate(self, request):
        prompt = request

        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step4_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()

        return generated_text

    def step5_generate(self, request, object_list):
        prompt = request + ' | ' + object_list + ' | '

        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step4_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()

        return generated_text

class ModifyModel(nn.Module):
    def __init__(self, 
                 MODEL_ID,
                 TOKENIZER_ID, 
                 adapter_layer1=None, 
                 adapter_layer2=None, 
                 adapter_layer3=None,
                 adapter_layer4=None,
                 adapter_layer5=None,
                 classify=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID,
                                                       model_max_length=4096,
                                                       use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", trust_remote_code=True
        )
        
        if (adapter_layer1 != None):
            self.step1_layer = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer1,
                                                        is_trainable=False)

        if (adapter_layer2 != None):
            self.step2_layer = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer2,
                                                        is_trainable=False)

        if (adapter_layer3 != None):
            self.step3_layer = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer3,
                                                        is_trainable=False)

        if (adapter_layer4 != None):
            self.adapter_layer4 = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer4,
                                                        is_trainable=False)

        if (adapter_layer5 != None):
            self.adapter_layer5 = PeftModel.from_pretrained(self.base_model,
                                                        adapter_layer5,
                                                        is_trainable=False)
        
        if (classify != None):
            self.classify_layer = PeftModel.from_pretrained(self.base_model,
                                                        classify,
                                                        is_trainable=False)
    
    def classify_generate(self, 
                          original_prompt,
                          modify_prompt):
        prompt = original_prompt + ' | ' + modify_prompt + ' | ' 
        
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')

        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step1_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        generated_text = output_text[len(prompt):].strip()

        return int(generated_text)        

    def step1_generate(self, 
                       original_prompt, 
                       original_object_list,
                       modify_prompt):
        prompt = original_prompt + ' | ' + original_object_list + ' | ' + modify_prompt + ' | ' 

        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')

        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step1_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()

        return generated_text

    def step2_generate(self,
                       original_prompt, 
                       original_init_pos,
                       modify_prompt):
        prompt = original_prompt + ' | ' + original_init_pos + ' | ' + modify_prompt + ' | '

        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step2_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()  

        return generated_text

    def step3_generate(self,
                       original_prompt, 
                       original_movements,
                       modify_prompt):
        prompt = original_prompt + ' | ' + original_movements + ' | ' + modify_prompt + ' | '

        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')

        tokenized_prompt = tokenized_prompt.to('cuda:0')

        output_ids = self.step3_layer.generate(**tokenized_prompt, max_new_tokens=1024)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip() 

        return generated_text