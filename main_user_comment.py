import json
from LanguageModel import ScenePlanningModel, TestScenePlanningModel

def step_1_running():
  pass

def step_2_running():
  pass

def step_3_running():
  pass

def step_4_running():
  pass

def step_5_running():
  pass

if __name__ == '__main__':
  
  # Open previous answer
  with open('step1_respone.json', 'r') as openfile:
    step1_respone = json.load(openfile)

  with open('step2_respone.json', 'r') as openfile:
    step2_respone = json.load(openfile)

  with open('step3_respone.json', 'r') as openfile:
    step3_respone = json.load(openfile)

  with open('step4_respone.json', 'r') as openfile:
    step4_respone = json.load(openfile)

  with open('step5_respone.json', 'r') as openfile:
    step5_respone = json.load(openfile)
  
  # Loading Model
  print("\n------------------------------------------------------")
  print("Loading Llama3 8B with adapter...")
  # MODEL_ID = "LoftQ/Meta-Llama-3-8B-4bit-64rank"
  # adapter_layers = []
  # scene_plan_model = ScenePlanningModel(MODEL_ID=MODEL_ID, adapter_layers=adapter_layers)
  scene_plan_model = TestScenePlanningModel()
  print("Done!")
  print("------------------------------------------------------")

  modify_command = input("Input your modified: ")

  respone = scene_plan_model.modified_user_request(batch=[modify_command],
                                          step1_respone=step1_respone,
                                          step2_respone=step2_respone,
                                          step3_respone=step3_respone,
                                          step4_respone=step4_respone,
                                          step5_respone=step5_respone)


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

  exec(respone[0])

  modify_step = min(change_step)

  if (modify_step == 1):
    pass
  if (modify_step == 2):
    pass
  if (modify_step == 3):
    pass
  if (modify_step == 4):
    pass
  if(modify_step == 5):
    pass