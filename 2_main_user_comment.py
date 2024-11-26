import json
from LanguageModel import ScenePlanningModel, TestScenePlanningModel

if __name__ == '__main__':  

    local_vars = {}
    print("------------------------------------------------------")
    print("Loading request...")
    with open('user_interact_result.json', 'r') as openfile:
        input_text = json.load(openfile)
    input_text = input_text["user_interact_result"]
    print("Done!")
    print("------------------------------------------------------")

    print("\n------------------------------------------------------")
    print("Open previous answer")
    with open('step1_respone.json', 'r') as openfile:
        step1_respone_previous = json.load(openfile)

    with open('step2_respone.json', 'r') as openfile:
        step2_respone_previous = json.load(openfile)

    with open('step3_respone.json', 'r') as openfile:
        step3_respone_previous = json.load(openfile)

    with open('step4_respone.json', 'r') as openfile:
        step4_respone_previous = json.load(openfile)

    with open('step5_respone.json', 'r') as openfile:
        step5_respone_previous = json.load(openfile)
    print("------------------------------------------------------")

    model_choice = input("Which model you want to use? (1 - GPT api, 2 - Llam3 8B): ")
    MODEL_ID = "LoftQ/Meta-Llama-3-8B-4bit-64rank"
    # Khởi tạo Llama3-8B-Quantization
    if (int(model_choice) == 1):
        print("------------------------------------------------------")
        print("Initialize GPT api...")
        # user_interact_model = UserInteractModel(MODEL_ID=MODEL_ID)
        scene_plan_model = TestScenePlanningModel()
        print("Done!")
        print("------------------------------------------------------")
    elif (int(model_choice) == 2):
        print("------------------------------------------------------")
        print("Initialize LoftQ/Meta-Llama-3-8B-4bit-64rank...")
        scene_plan_model = ScenePlanningModel(MODEL_ID=MODEL_ID)
        print("Done!")
        print("------------------------------------------------------")

    modify_command = input("Input your modification: ")

    respone = scene_plan_model.modified_user_request(batch=[modify_command],
                                            step1_respone=step1_respone_previous,
                                            step2_respone=step2_respone_previous,
                                            step3_respone=step3_respone_previous,
                                            step4_respone=step4_respone_previous,
                                            step5_respone=step5_respone_previous)


    # Viết ra file .json và yêu cầu người dùng check lại
    json_object = json.dumps(respone, indent=4)
    with open("modify_step.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in modify_step.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')

    # Đọc lại file .json để đến bước tiếp theo
    with open('modify_step.json', 'r') as openfile:
        respone = json.load(openfile)
    print("------------------------------------------------------")

    exec(respone[0], {}, local_vars)

    modify_step = min(local_vars['change_step'])

    if (modify_step == 1):
        # Phase 1: Xác định các vật thể chính
        print("\n------------------------------------------------------")
        print("Xác định các vật thể chính")
        step1_respone = scene_plan_model.step1_generate(batch=input_text, 
                                                        mode="modify",
                                                        feedback=modify_command, 
                                                        previous_answers=step1_respone_previous)
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step1_respone, indent=4)
        with open("step1_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step1_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step1_respone.json', 'r') as openfile:
            step1_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step1_respone[0])

        # Update respone
        input_text = scene_plan_model.updatePrompt(batch=input_text, 
                                                   batchComand=[modify_command])

        print("\n------------------------------------------------------")
        print("Xác định hướng và vị trí ban đầu của các vật thể")
        step2_respone = scene_plan_model.step2_generate(batch=input_text, objects_list=step1_respone[0], mode="new")
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step2_respone, indent=4)
        with open("step2_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step2_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step2_respone.json', 'r') as openfile:
            step2_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step2_respone[0])   

        # Sinh ra hành động và chuyển động của các vật thể
        print("\n------------------------------------------------------")
        print("Sinh ra hành động và chuyển động của các vật thể")
        step3_respone = scene_plan_model.step3_generate(batch=input_text, 
                                                        objects_list=step1_respone[0], 
                                                        init_pos_ori=step2_respone[0],
                                                        mode="new")
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step3_respone, indent=4)
        with open("step3_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step3_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step3_respone.json', 'r') as openfile:
            step3_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step3_respone[0]) 
    if (modify_step == 2):
        print("\n------------------------------------------------------")
        print("Xác định hướng và vị trí ban đầu của các vật thể")
        step2_respone = scene_plan_model.step2_generate(batch=input_text, 
                                                        objects_list=step1_respone_previous, 
                                                        mode="modify",
                                                        feedback=modify_command,
                                                        previous_answers=step2_respone_previous)
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step2_respone, indent=4)
        with open("step2_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step2_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step2_respone.json', 'r') as openfile:
            step2_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step2_respone[0])   

        # Update respone
        input_text = scene_plan_model.updatePrompt(batch=input_text, 
                                                   batchComand=[modify_command])

        # Sinh ra hành động và chuyển động của các vật thể
        print("\n------------------------------------------------------")
        print("Sinh ra hành động và chuyển động của các vật thể")
        step3_respone = scene_plan_model.step3_generate(batch=input_text, 
                                                        objects_list=step1_respone_previous, 
                                                        init_pos_ori=step2_respone[0],
                                                        mode="new")
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step3_respone, indent=4)
        with open("step3_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step3_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step3_respone.json', 'r') as openfile:
            step3_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step3_respone[0]) 

    if (modify_step == 3):
            # Sinh ra hành động và chuyển động của các vật thể
        print("\n------------------------------------------------------")
        print("Sinh ra hành động và chuyển động của các vật thể")
        step3_respone = scene_plan_model.step3_generate(batch=input_text, 
                                                        objects_list=step1_respone_previous, 
                                                        init_pos_ori=step2_respone_previous,
                                                        mode="modify",
                                                        feedback=modify_command,
                                                        previous_answers=step3_respone_previous)
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step3_respone, indent=4)
        with open("step3_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step3_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step3_respone.json', 'r') as openfile:
            step3_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step3_respone[0]) 

        # Update respone
        input_text = scene_plan_model.updatePrompt(batch=input_text, 
                                                   batchComand=[modify_command])

    if (modify_step == 4):
        # Xác định các vật thể trong môi trường
        print("\n------------------------------------------------------")
        print("Xác định các vật thể trong môi trường")
        step4_respone = scene_plan_model.step4_generate(batch=input_text, 
                                                        main_object=step1_respone_previous, 
                                                        mode="modify",
                                                        feedback=modify_command,
                                                        previous_answers=step4_respone_previous)
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step4_respone, indent=4)
        with open("step4_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step4_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step4_respone.json', 'r') as openfile:
            step4_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step4_respone[0]) 

        # Update respone
        input_text = scene_plan_model.updatePrompt(batch=input_text, 
                                                   batchComand=[modify_command])

        # Xác định vị trí và các ràng buộc giữa các vật thể 
        print("\n------------------------------------------------------")
        print("Generate the motion script of the main objects.")
        step5_respone = scene_plan_model.step5_generate(batch=input_text, 
                                                        object_list=step4_respone[0],
                                                        mode="new")
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step5_respone, indent=4)
        with open("step5_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step5_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step5_respone.json', 'r') as openfile:
            step5_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step5_respone[0]) 
    if(modify_step == 5):
        # Xác định vị trí và các ràng buộc giữa các vật thể 
        print("\n------------------------------------------------------")
        print("Generate the motion script of the main objects.")
        step5_respone = scene_plan_model.step5_generate(batch=input_text, 
                                                        object_list=step4_respone_previous,
                                                        mode="modify",
                                                        feedback=modify_command,
                                                        previous_answers=step5_respone_previous)
        print("Done!")
        print("------------------------------------------------------")

        # Viết ra file .json và yêu cầu người dùng check lại
        json_object = json.dumps(step5_respone, indent=4)
        with open("step5_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step5_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        # Đọc lại file .json để đến bước tiếp theo
        with open('step5_respone.json', 'r') as openfile:
            step5_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step5_respone[0]) 
        
        # Update respone
        input_text = scene_plan_model.updatePrompt(batch=input_text, 
                                                   batchComand=[modify_command])