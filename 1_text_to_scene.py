import json

from LanguageModel import ScenePlanningModel

if __name__ == '__main__':
    # Đọc respone từ user_interact_result.json
    print("------------------------------------------------------")
    print("Loading request...")
    with open('user_interact_result.json', 'r') as openfile:
        input_text = json.load(openfile)
    input_text = input_text["user_interact_result"][0]
    print("Done!")
    print("------------------------------------------------------")

    MODEL_ID = "apple/OpenELM-270M"
    TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"
    adapter_layer1='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step1_LoRA_TextToScene', 
    adapter_layer2='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step2_LoRA_TextToScene', 
    adapter_layer3='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step3_LoRA_TextToScene',
    adapter_layer4='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step4_LoRA_TextToScene',
    adapter_layer5='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step5_LoRA_TextToScene'
    print("------------------------------------------------------")
    print("Initialize OpenELM-270M...")
    scene_plan_model = ScenePlanningModel(MODEL_ID=MODEL_ID, 
                                        TOKENIZER_ID=TOKENIZER_ID,
                                        adapter_layer1=adapter_layer1,
                                        adapter_layer2=adapter_layer2,
                                        adapter_layer3=adapter_layer3,
                                        adapter_layer4=adapter_layer4,
                                        adapter_layer5=adapter_layer5)
    print("Done!")
    print("------------------------------------------------------")

    print("\n------------------------------------------------------")
    print("Identify key objects")
    step1_respone = [scene_plan_model.step1_generate(request=input_text)]
    print("Done!")
    print("------------------------------------------------------")

    json_object = json.dumps(step1_respone, indent=4)
    with open("step1_respone.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in step1_respone.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    with open('step1_respone.json', 'r') as openfile:
        step1_respone = json.load(openfile)
    print("------------------------------------------------------")
    exec(step1_respone[0])

    print("\n------------------------------------------------------")
    print("Determine the initial position of the objects")
    step2_respone = [scene_plan_model.step2_generate(request=input_text, 
                                                    objects_list=object_list)]
    print("Done!")
    print("------------------------------------------------------")

    json_object = json.dumps(step2_respone, indent=4)
    with open("step2_respone.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in step2_respone.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    with open('step2_respone.json', 'r') as openfile:
        step2_respone = json.load(openfile)
    print("------------------------------------------------------")
    exec(step2_respone[0])   

    print("\n------------------------------------------------------")
    print("Generates actions and movements of objects")
    step3_respone = [scene_plan_model.step3_generate(request=input_text, 
                                                    objects_list=object_list, 
                                                    init_pos_ori=init_pos)]
    print("Done!")
    print("------------------------------------------------------")

    json_object = json.dumps(step3_respone, indent=4)
    with open("step3_respone.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in step3_respone.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    with open('step3_respone.json', 'r') as openfile:
        step3_respone = json.load(openfile)
    print("------------------------------------------------------")
    exec(step3_respone[0]) 

    print("\n------------------------------------------------------")
    print("Identify objects creating the environment")
    step4_respone = [scene_plan_model.step4_generate(request=input_text)]
    print("Done!")
    print("------------------------------------------------------")

    json_object = json.dumps(step4_respone, indent=4)
    with open("step4_respone.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in step4_respone.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    with open('step4_respone.json', 'r') as openfile:
        step4_respone = json.load(openfile)
    print("------------------------------------------------------")
    exec(step4_respone[0]) 

    print("\n------------------------------------------------------")
    print("Generate the motion script of the key objects.")
    step5_respone = [scene_plan_model.step5_generate(request=input_text, 
                                                    object_list=object_evironment_list)]
    print("Done!")
    print("------------------------------------------------------")

    json_object = json.dumps(step5_respone, indent=4)
    with open("step5_respone.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in step5_respone.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    with open('step5_respone.json', 'r') as openfile:
        step5_respone = json.load(openfile)
    print("------------------------------------------------------")
    exec(step5_respone[0])