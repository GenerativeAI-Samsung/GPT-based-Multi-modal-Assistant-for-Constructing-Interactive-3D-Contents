import json
import os

from LanguageModel import ScenePlanningModel, TestScenePlanningModel

def evaluate(scene_plan_model, text, index, dataType, modelType):
    print(f"Start index {index}")
    # Phase 1: Xác định các vật thể chính
    print("\n\t------------------------------------------------------")
    print("\tXác định các vật thể chính")
    step1_respone = scene_plan_model.step1_generate(batch=[text], mode="new")
    print("\tDone!")
    print("------------------------------------------------------")

    print("\n------------------------------------------------------")
    print("\tXác định hướng và vị trí ban đầu của các vật thể")
    step2_respone = scene_plan_model.step2_generate(batch=[text], objects_list=step1_respone[0], mode="new")
    print("\tDone!")
    print("------------------------------------------------------")
    
    # Sinh ra hành động và chuyển động của các vật thể
    print("\n------------------------------------------------------")
    print("\tSinh ra hành động và chuyển động của các vật thể")
    step3_respone = scene_plan_model.step3_generate(batch=[text], 
                                                    objects_list=step1_respone[0], 
                                                    init_pos_ori=step2_respone[0],
                                                    mode="new")
    print("\tDone!")
    print("\t------------------------------------------------------")

    # Xác định các vật thể trong môi trường
    print("\n\t------------------------------------------------------")
    print("\tXác định các vật thể trong môi trường")
    step4_respone = scene_plan_model.step4_generate(batch=[text], 
                                                    main_object=step1_respone[0], 
                                                    mode="new")
    print("\tDone!")
    print("\t------------------------------------------------------")

    # Xác định vị trí và các ràng buộc giữa các vật thể 
    print("\n\t------------------------------------------------------")
    print("\tGenerate the motion script of the main objects.")
    step5_respone = scene_plan_model.step5_generate(batch=[text], 
                                                    object_list=step4_respone[0],
                                                    mode="new")
    print("\tDone!")
    print("\t------------------------------------------------------")

    folderPath = f'/content/{modelType}/{dataType}/'

    # Check if parent folder
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    os.makedirs(folderPath + f'sample{index}/')
    # Write output to folder 
    with open(folderPath + f'sample{index}/' + 'step1.json', 'w') as json_file:
        json.dump(step1_respone, json_file, indent=4)
    with open(folderPath + f'sample{index}/' + 'step2.json', 'w') as json_file:
        json.dump(step2_respone, json_file, indent=4)
    with open(folderPath + f'sample{index}/' + 'step3.json', 'w') as json_file:
            json.dump(step3_respone, json_file, indent=4)
    with open(folderPath + f'sample{index}/' + 'step4.json', 'w') as json_file:
            json.dump(step4_respone, json_file, indent=4)
    with open(folderPath + f'sample{index}/' + 'step5.json', 'w') as json_file:
            json.dump(step5_respone, json_file, indent=4)

if __name__ == '__main__':
    modelType = input("Model you want evaluate (GPT or Llama3)?")
    dataType = input("Type of samples (easy, medium or hard)?: ")
    
    print("Loading data...")
    if (dataType == "easy"):
        with open('/content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/easy_samples.js', 'r') as file:
            data = json.load(file)
    elif (dataType == "medium"):
        with open('/content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/medium_samples.js', 'r') as file:
            data = json.load(file)
    elif ((dataType == "hard")):
        with open('/content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/hard_samples.js', 'r') as file:
            data = json.load(file)
    print(f"Total samples: {len(data)}")

    print("Loading model...")
    if (modelType == "GPT"):
        print("Initialize GPT api...")
        scene_plan_model = TestScenePlanningModel()
        print("Done!")
        print("------------------------------------------------------")
    elif (modelType == "Llama3"):
        print("------------------------------------------------------")
        print("Initialize LoftQ/Meta-Llama-3-8B-4bit-64rank...")
        MODEL_ID = "LoftQ/Meta-Llama-3-8B-4bit-64rank"
        scene_plan_model = ScenePlanningModel(MODEL_ID=MODEL_ID)
        print("Done!")
        print("------------------------------------------------------")

    indexStart = input("Index start: ")
    indexEnd = input("Index end: ")

    for i in range(indexStart, indexEnd + 1):
         evaluate(scene_plan_model=scene_plan_model,
                  text=data[i],
                  index=i,
                  dataType=dataType,
                  modelType=modelType)