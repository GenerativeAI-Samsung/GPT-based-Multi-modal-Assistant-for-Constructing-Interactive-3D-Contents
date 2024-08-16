import json
import random

import torch
import torch.optim as optim
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from typing import Dict

from Step1 import step1
from Step2 import step2
from Step3 import step3
from Step4 import step4
from Step5 import step5

from utils import generate_reward_score_from_api, interact_with_lm, RLAIF_loss_fuction

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    base_model: transformers.PreTrainedModel
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    base_model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def prompt_reward(criteria, answer_format, prompt, response):
    formatted_criteria = "".join(f"\t-{item['name']}: {item['description']}\n" for item in criteria)
    rewarding_prompt = f"""
You are an evaluator. Your task is to grade the response provided by the responder to the user's request based on specific criteria, using a 100-point scale.
The criteria include:
{formatted_criteria}

The responder's answer is formatted as:
{answer_format}

After determining your answer, structure them in this format:
rewarding_score = [{{"name": criteria1, "score": score1, "description": description1}}, 
                    {{"name": criteria2, "score": score2, "description": description2}},
                    ...]

Avoid using normal text; format your response strictly as specified above.
----------------------------------------------------------------------------------------------------------
User's request: "{prompt}"

Responder's answer: {response}
"""
    return rewarding_prompt

def running_step1(tokenizer, model, base_model, criteria, user_request):
    step1_answer_format, step1_prompt, step1_response, step1_last_hidden_state = step1(tokenizer=tokenizer, model=model, user_request=user_request)
    
    _, step1_base_last_hidden_state = interact_with_lm(tokenizer=tokenizer, model=base_model, prompt=step1_prompt, setting="base_model")
    rewarding_prompt = prompt_reward(criteria=criteria, answer_format=step1_answer_format, prompt=step1_prompt, response=step1_response)
    return rewarding_prompt, step1_last_hidden_state, step1_base_last_hidden_state

def running_step2(tokenizer, model, criteria, user_request):
    _, _, step1_response, _ = step1(tokenizer=tokenizer, model=model, user_request=user_request)
    step2_answer_format, step2_prompt, step2_response, step2_last_hidden_state = step2(tokenizer=tokenizer, model=model, user_request=user_request, step1_respone=step1_response)

    _, step2_base_last_hidden_state = interact_with_lm(tokenizer=tokenizer, model=base_model, prompt=step2_prompt, setting="base_model")
    rewarding_prompt = prompt_reward(criteria=criteria, answer_format=step2_answer_format, prompt=step2_prompt, response=step2_response)
    return rewarding_prompt, step2_last_hidden_state, step2_base_last_hidden_state

def running_step3(tokenizer, model, criteria, user_request):
    _, _, step1_response, _ = step1(tokenizer=tokenizer, model=model, user_request=user_request)
    step3_answer_format, step3_prompt, step3_response, step3_last_hidden_state = step3(tokenizer=tokenizer, model=model, user_request=user_request, step1_respone=step1_response)

    _, step3_base_last_hidden_state = interact_with_lm(tokenizer=tokenizer, model=base_model, prompt=step3_prompt, setting="base_model")
    rewarding_prompt = prompt_reward(criteria=criteria, answer_format=step3_answer_format, prompt=step3_prompt, response=step3_response)
    return rewarding_prompt, step3_last_hidden_state, step3_base_last_hidden_state

def running_step4(tokenizer, model, criteria, user_request):
    _, _, step1_response, _ = step1(tokenizer=tokenizer, model=model, user_request=user_request)
    _, _, step3_response, _ = step3(tokenizer=tokenizer, model=model, user_request=user_request, step1_respone=step1_response)
    step4_answer_format, step4_prompt, step4_response, step4_last_hidden_state = step4(tokenizer=tokenizer, model=model, user_request=user_request, step1_respone=step1_response, step3_respone=step3_response)

    _, step4_base_last_hidden_state = interact_with_lm(tokenizer=tokenizer, model=base_model, prompt=step4_prompt, setting="base_model")
    rewarding_prompt = prompt_reward(criteria=criteria, answer_format=step4_answer_format, prompt=step4_prompt, response=step4_response)
    return rewarding_prompt, step4_last_hidden_state, step4_base_last_hidden_state

def running_step5(tokenizer, model, criteria, user_request):
    _, _, step1_response, _ = step1(tokenizer=tokenizer, model=model, user_request=user_request)
    _, _, step2_response, _ = step2(tokenizer=tokenizer, model=model, user_request=user_request, step1_respone=step1_response)
    _, _, step3_response, _ = step3(tokenizer=tokenizer, model=model, user_request=user_request, step1_respone=step1_response)
    _, _, step4_response, _ = step4(tokenizer=tokenizer, model=model, user_request=user_request, step1_respone=step1_response, step3_respone=step3_response)

    exec(step2_response)
    exec(step4_response)

    main_obj_initial_position = {}

    for main_obj in main_objs:
        main_obj_initial_position[main_obj] = initial_position[main_obj]

    step5_answer_format, step5_prompt, step5_response, step5_last_hidden_state = step5(tokenizer=tokenizer, model=model, user_request=user_request, step3_response=step3_response, initial_position=main_obj_initial_position)

    _, step5_base_last_hidden_state = interact_with_lm(tokenizer=tokenizer, model=base_model, prompt=step5_prompt, setting="base_model")
    rewarding_prompt = prompt_reward(criteria=criteria, answer_format=step5_answer_format, prompt=step5_prompt, response=step5_response)
    return rewarding_prompt, step5_last_hidden_state, step5_base_last_hidden_state

def save_model(model, diretory="/content/adapter_folder"):
    model.save_pretrained(diretory)
    print('saving model...')

def train(tokenizer, 
          model, 
          base_model, 
          criterias, 
          running_step,
          train_data_path,
          num_epoch, 
          batch_size,
          learning_rate,
          shuffle=True):
    
    # train_data_path is a .json file contains a list where each item is a user's request
    # load train_data
    # Open json file
    print("Loading data...")
    f = open(train_data_path)

    # load json object
    train_data = json.load(f)

    # Load history .json
    print("Loading history file...")
    f = open(train_data_path)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    print("Start training...")
    for epoch in range(num_epoch):

        # Shuffle index data list
        if (shuffle == True):
            index_list = [i for i in range(len(train_data))]
            index_list = random.shuffle(index_list)
        else:
            index_list = [i for i in range(len(train_data))]
        
        num_batch = len(train_data) % batch_size
        for i in range(num_batch):
            optimizer.zero_grad()

            batch_data = []
            for j in range(j * i, j * i + batch_size):
                batch_data.append(train_data[j])

            # Get response from Llama3 and feedback from GPT-4
            custom_run = f"rewarding_prompt, last_hidden_state, base_last_hidden_state = running_step{running_step}(tokenizer=tokenizer, model=model, base_model=base_model, criteria=criterias, user_request=batch_data)"
            exec(custom_run)

            score_response = generate_reward_score_from_api(prompt=rewarding_prompt)
            exec(score_response)

            # Caculate loss 
            loss_value = RLAIF_loss_fuction(rewarding_score=rewarding_score, last_hidden_state=last_hidden_state, base_last_hidden_state=base_last_hidden_state)

            loss_value.backward()
            optimizer.step()

            print(f"epoch: {epoch}, batch: {i}")
        evaluate_loss = evaluate(tokenizer=tokenizer, model=model, base_model=base_model, criterias=criterias, running_step=running_step, evaluate_data_path='/content/evalue_data', batch_size=4)

    
    # Save model
    save_model(model)

def evaluate(tokenizer, 
          model, 
          base_model, 
          criterias, 
          running_step,
          evaluate_data_path, 
          batch_size):

    # Load evaluate data    
    # Open json file
    print("Loading data...")
    f = open(evaluate_data_path)

    # load json object
    evaluate_data = json.load(f)

    print("Start evaluate...")
    
    final_loss = 0

    num_batch = len(evaluate_data) % batch_size
    for i in range(num_batch):
        batch_data = []
        for j in range(j * i, j * i + batch_size):
            batch_data.append(evaluate_data[j])

        # Get response from Llama3 and feedback from GPT-4
        custom_run = f"rewarding_prompt, last_hidden_state, base_last_hidden_state = running_step{running_step}(tokenizer=tokenizer, model=model, base_model=base_model, criteria=criterias, user_request=batch_data)"
        exec(custom_run)

        score_response = generate_reward_score_from_api(prompt=rewarding_prompt)
        exec(score_response)

        # Caculate loss 
        loss_value = RLAIF_loss_fuction(rewarding_score=rewarding_score, last_hidden_state=last_hidden_state, base_last_hidden_state=base_last_hidden_state)

        final_loss += loss_value
    
    final_loss = final_loss / num_batch
    return final_loss

if __name__ == '__main__':
    # Interface
    print('Please choose which step you want to work with:\n')
    print('\tstep1: Identify the objects that will appear in the 3D scene.\n')
    print('\tstep2: Classify the types of the identified objects.\n')
    print('\tstep3: Generate a general description of the layout.\n')
    print('\tstep4: Generate the initial location coordinates of the objects and the constraints between them\n')
    print('\tstep5: Generate the motion script of the main objects.\n')
    running_step = input('Running Step (1 to 5): ')
    print('---------------------------------------------------------------\n')
    print('Please choose setting option \n')
    print('\t1: Fine-tune from scratch\n')
    print('\t2: Resume fine-tune')
    print('\t3: Evaluate')
    setting_option = input('Setting Option (1 to 3): ')
    print('---------------------------------------------------------------\n')
    print('Running...')

    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    step1_criteria = [
        {'name': 'Accuracy',
            'description': 'Are the objects identified fully and accurately?'},
        {'name': 'Coverage',
            'description': 'Is any necessary object for the scene missing?'},
        {'name': 'Relevance to Requirements',
            'description': 'Do the objects align with the goals and requirements of the scene?'}
    ]

    step2_criteria = [
        {'name': 'Classification Accuracy',
            'description': 'Are the objects correctly classified'},
        {'name': 'Consistency',
            'description': 'Are the object types classified consistently?'},
        {'name': 'Scalability',
            'description': 'Is the classification system scalable to include new object types?'},
        {'name': 'Contextual Relevance',
            'description': 'Does the object type classification fit the overall context of the scene?'},
        {'name': 'Object Hierarchy Accuracy',
            'description': 'Are the objects accurately classified according to their importance or functionality in the scene?'},
        {'name': 'Contextual Classification',
            'description': 'Does the classification consider the specific context of the scene, or is it based only on general characteristics?'}
    ]

    step3_criteria = [
        {'name': 'Clarity and Detail',
            'description': 'Is the description clear and detailed enough to visualize the scene layout?'},
        {'name': 'Logical Layout',
            'description': 'Is the layout logical and natural within the context?'},
        {'name': 'Consistency',
            'description': 'Are the descriptions consistent across different runs of the pipeline?'},
        {'name': 'Design Requirement Alignment',
            'description': 'Does the layout description align with the initial design requirements?'},
        {'name': 'Modularity of Layout',
            'description': 'Can the layout be easily adjusted or expanded without affecting the overall structure?'},
        {'name': 'Emotional Conveyance',
            'description': 'Does the layout convey the emotions needed for the scene, such as joy, tranquility, or tension?'},
        {'name': 'Aesthetic Quality',
            'description': 'Does the layout meet aesthetic requirements, such as balance, harmony, and visual appeal?'}
    ]

    step4_criteria = [
        {'name': 'Coordinate Accuracy',
            'description': 'Are the object coordinates accurate?'},
        {'name': 'Relationship Logic',
            'description': 'Are the constraints between objects logical and aligned with the layout description?'},
        {'name': 'Consistency',
            'description': 'Are the coordinates and constraints consistent across different runs?'},
        {'name': 'Real-world Suitability',
            'description': 'Are the coordinates and constraints practical for real-world implementation of the scene?'},
        {'name': 'Constraint Strength',
            'description': 'Are the constraints based on the objects\' physical interaction and load-bearing capacity?'},
        {'name': 'Dynamic Adjustability',
            'description': 'Are the coordinates easily adjustable to accommodate layout changes?'},
        {'name': 'Environmental Response',
            'description': 'Do the coordinates and constraints consider environmental factors such as lighting, wind, and weather?'}
    ]

    step5_criteria = [
        {'name': 'Smoothness of Movement',
            'description': 'Are the movements smooth and continuous?'},
        {'name': 'Logical Movement',
            'description': 'Are the movements logical and natural?'},
        {'name': 'Consistency',
            'description': 'Are the movements consistent and free from distortion across different runs?'},
        {'name': 'Script Alignment',
            'description': 'Do the movements align with the script and the overall context of the scene?'},
        {'name': 'Interaction of Movements',
            'description': 'Do the movements interact well with the environment and other objects?'},
        {'name': 'Adaptive Movement',
            'description': 'Can the movements be easily adjusted to fit changes in the script or environment?'},
        {'name': 'Uniqueness of Movement',
            'description': 'Are the movements unique, creative, and do they help highlight the main character?'},
        {'name': 'Dynamic Effects',
            'description': 'Do the movements generate physical effects such as collisions, resistance, or force impact?'}
    ]

    # Access to choosen criteria 
    criterias = [step1_criteria, step2_criteria, step3_criteria, step4_criteria, step5_criteria]
    working_criteria = criterias[int(running_step) - 1]

    # Loading model with setting (Default: Meta-Llama-3-8B-4bit-64rank)
    adapter_folder = f"/content/adapter_folder/step{int(running_step)}" 

    # Model and tokenizer IDs
    MODEL_ID = "LoftQ/Meta-Llama-3-8B-4bit-64rank"

    # Loading base model
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    if (setting_option == '1'): 
        peft_model = PeftModel.from_pretrained(
                base_model,
                MODEL_ID,
                subfolder="loftq_init",
                is_trainable=True)
    elif (setting_option == '2'):
        peft_model = PeftModel.from_pretrained(
                base_model,
                adapter_folder,
                is_trainable=True)
    elif (setting_option == '3'):
        peft_model = PeftModel.from_pretrained(
                base_model,
                adapter_folder,
                is_trainable=False)
    
    # Loading tokenizer
    # model_max_length = 512 by default
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )

    # Adding special token to tokenizer
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=peft_model,
        base_model=base_model
    )

    if (setting_option == '1' or setting_option == '2'):
        train(tokenizer=tokenizer, 
              model=peft_model, 
              base_model=base_model, 
              criterias=working_criteria, 
              running_step=running_step,
              train_data_path='/content/train_data',
              history_path = '/content/history',
              num_epoch=5,
              batch_size=4,
              learning_rate=1e-7,
              shuffle=True)
    elif (setting_option == '3'):
        evaluate()

    # Test
#     user_request = """
# To create a 3D scene for the text "A girl plays with her dog in the garden," we can use the following details:

#     The Setting:
#         The scene takes place in a garden, which includes a grassy field.
#         Consider adding elements to make it look like a children's playground.

#     The Girl:
#         She is young, possibly a little girl or a pre-teen.
#         She is wearing a yellow sweater, a dress, and a pair of high heels.
#         Her clothing is mostly bright and colorful.

#     The Dog:
#         The dog is medium-sized.
#         The dog can be portrayed standing or interacting with the girl.

#     Interaction:
#         The girl is playing with the dog, so they should be interacting with each other in a playful manner.

# Based on these details, you can imagine a vibrant and joyful scene. 
# The girl, dressed in her colorful outfit, is happily playing with her medium-sized dog in a lush, green garden. 
# The garden might include elements of a playground to enhance the playful atmosphere.
# """
    
#     # Get response from Llama3 and feedback from GPT-4
#     custom_run = f"rewarding_prompt, last_hidden_state, base_last_hidden_state = running_step{running_step}(tokenizer=tokenizer, model=peft_model, base_model=base_model, criteria=working_criteria, user_request=user_request)"
#     exec(custom_run)

#     score_response = generate_reward_score_from_api(prompt=rewarding_prompt)
#     exec(score_response)

#     loss_value = RLAIF_loss_fuction(rewarding_score=rewarding_score, last_hidden_state=last_hidden_state, base_last_hidden_state=base_last_hidden_state)

#     print(f"reward_prompt: {rewarding_prompt}")
#     print("\n--------------------------------------------------------------\n")
#     print(f"rewarding_score: {rewarding_score}")
#     print("\n--------------------------------------------------------------\n")
#     print(f"loss_value: {loss_value}")
