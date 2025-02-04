from RAG import RAG_module
from LanguageModel import VisionLangugeModel, UserInteractModel
import json
import torch
import gc

# Input Text: "The scene happens in the garden. I want to use the objects described in the image. There are two such objects, and their initial positions are random. The first object runs in an ellipse, while the second one runs straight."

# Image Description: "I want the objects in the scene to be the objects in the image."
if __name__ == '__main__':

    input_text = input("Please provide your command: ")

    external_texts = input("Please provide external text: ")

    external_images = []
    print("Please provide image path and description")
    control = None
    while (control != 'done'):
        temp_image_path = input("Please provide image_path: ")
        temp_image_description = input("Please provide image description: ")
        external_images.append({"image_path": temp_image_path, 
               "image_description": temp_image_description,
               "image_questions": None,
               "questions_response": None})
        control = input("Are you done yet? (press 'done' if done, else press 'no'): ")

    MODEL_ID = "LoftQ/Meta-Llama-3-8B-Instruct-4bit-64rank"

    print("------------------------------------------------------")
    print("Initialize LoftQ/Meta-Llama-3-8B-Instruct-4bit-64rank...")
    user_interact_model = UserInteractModel(MODEL_ID=MODEL_ID)
    print("Done!")
    print("------------------------------------------------------")

    documents_augement = [external_texts]
    
    print("\n------------------------------------------------------")
    print('User Interact Model is asking questions about image...')
    question_asked = []
    for i, image in enumerate(external_images):
        prompt = f"""
You are an assistant that helps answer user requests.
The user provides you with some pictures, and you have to respond to their request based on those pictures.
However, you do not have direct access to the pictures. The only way to approach them is by asking questions related to the pictures to gather the necessary information.
Your task is to create questions based on the description about the image to extract the needed information to fulfill the user's request.

Description: "{image["image_description"]}"

After getting the answers, format them as follows:
external_images[{i}]["image_questions"] = [question1, question2, ...]

Avoid using normal text; format your response strictly as specified above.
""" 

        prompt += f"""
        -------------------------------------------------------------------------
        REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
        external_images[{i}]["image_questions"] = [question1, question2, ...]
        ------------------------------------------------------------------------
        """
        prompt += "\nRespone:" 

        respone = user_interact_model.generate(batch=[prompt])
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = user_interact_model.generate(batch=[prompt])

        # Crop Response
        cropped_respone_batch = []
        for res in respone:
            temp1 = res.split('\nRespone:', 1)[1]
            if (f'external_images[{i}]["image_questions"] = [' in temp1):
                temp2 = temp1.split(f'external_images[{i}]["image_questions"] = [')[1]
                temp3 = temp2.split(']')[0]
                temp = f'external_images[{i}][\'image_questions\'] = [' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append(f'external_images[{i}]["image_questions"] = []')
                print(f'external_images[{i}]["image_questions"] = []')   
            
        question_asked.append({"index": i, "prompt": prompt, "respone": cropped_respone_batch[0]})
    print("Done!")
    print("------------------------------------------------------")

    print("\n------------------------------------------------------")
    json_object = json.dumps(question_asked, indent=4)
    with open("question_asked.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in question_asked.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    with open('question_asked.json', 'r') as openfile:
        question_asked = json.load(openfile)
    print("------------------------------------------------------")

    for item in question_asked:
        exec(item["respone"])
    
    del user_interact_model
    gc.collect()
    torch.cuda.empty_cache()

    print("------------------------------------------------------")
    print("Initialize MC-LLaVA-3b...")
    Vision_LM = VisionLangugeModel()
    print("Done!")
    print("------------------------------------------------------")

    print("\n------------------------------------------------------")
    print("Vision Language answer questions about image")

    answers = []
    for i, image in enumerate(external_images):
        temp_list = []
        for question in image["image_questions"]:
            prompt = f"""
Based on the image and the description of the picture below, please answer the following questions related to the image
Some question may not relevant to the image and description. In that case, you should answer "Unknown"
Description of the image: {image["image_description"]}
Question: {question}
"""
            respone = Vision_LM.process(image_path=image["image_path"],query=prompt)
            temp_list.append({"image_index": i, "question": question, "respone": respone})
        answers.append(temp_list)
    
    print("\n------------------------------------------------------")
    json_object = json.dumps(answers, indent=4)
    with open("answers.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in answers.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    with open('answers.json', 'r') as openfile:
        answers = json.load(openfile)
    print("------------------------------------------------------")

    for i, image in enumerate(external_images):
        image["questions_response"] = [item["respone"] for item in answers[i]]

    for image in external_images:
        documents_augement.extend(respone  for respone in image["questions_response"])

    print(documents_augement)

    del Vision_LM
    gc.collect()
    torch.cuda.empty_cache()

    print("\n------------------------------------------------------")
    print("Initialize RAG Module") 
    Retrieval_module = RAG_module()
    Retrieval_module.initialize_embedding_database(text=documents_augement)

    print("------------------------------------------------------")

    print("Start User Interact Interface...")

    print("------------------------------------------------------")
    print("Initialize LoftQ/Meta-Llama-3-8B-Instruct-4bit-64rank...")
    user_interact_model = UserInteractModel(MODEL_ID=MODEL_ID)
    print("Done!")
    print("------------------------------------------------------")


    initialPrompt = False
    respone = ""
    while (input_text != 'done'):

        output_RAG = Retrieval_module.find_top_k_embedding(query=input_text, k=20)
        combine_item = ''.join((item + "\n") for item in output_RAG)
        
        promptRequest = f"""
Describe a 3D scene in a single, continuous paragraph. The description must include:
    1.Location: Where the scene takes place.
    2.Initial Placement of Objects: The starting position of all necessary objects in the scene.
    3.Object Movements: Movements taken from the following list, ensuring each movement includes the required parameters:
        1. zigZag: Parameters include starting and ending coordinates.
        2. standStill: No parameters.
        3. runStraight: Parameters include starting and ending coordinates.
        4. ellipse: Parameters include starting coordinates, center coordinates, major axis length, and minor axis length.
        5. jump: Parameters include starting coordinates, peak coordinates, and ending coordinates.
        6. Linear Bézier Curve: Parameters include two points (P0 and P1).
        7. Quadratic Bézier Curve: Parameters include three points (P0, P1, P2).
        8. Cubic Bézier Curve: Parameters include four points (P0, P1, P2, P3).
    4. Frame Range: The starting and ending frame for each object's motion.
Ensure the description flows naturally and avoids numbered sections or headers. It must integrate all required details seamlessly. Below is an example of an expected response:
"The scene unfolds in a playground. A bicycle1 moves along a linear Bézier curve, starting at (3.62, 5.26, 18.24) and transitioning to (20.34, -27.23, 20.94), creating a simple path between frame 150 and frame 181. Meanwhile, giraffe1 runs straight along a path from (16.5, 21.26, -15.67) to (-32.88, 20.21, -4.08), quickly and steadily over frames 53 to 82. At the same time, giraffe1 jumps in a winding motion, bursting from (-8.93, -14.32, 25.71), accelerating toward (9.71, -2.13, 27.2) from frames 84 to 167. Horse1 jumps calmly and steadily, remaining poised in place without motion between frames 43 and 130, while giraffe1 stands still following a snaking course, starting at (-27.22, -18.32, 5.72) and darting toward (29.08, -10.61, 11.8) between frames 70 and 98."
Ensure your response strictly adheres to this format and incorporates all required details."""
        if (initialPrompt == False):
            promptRequest += f"""
Additionally, there is some supplementary information that will help you respond more accurately to the user's needs:
{combine_item}

"""
        elif (initialPrompt == True):
            promptRequest += """
Some information might conflict. Howerver, you should always priority what in User input
"""
        promptRequest += "\nRespone:" 
        print(f"prompt: {prompt}")
        print("responing...")

        respone = user_interact_model.generate(batch=[promptRequest])
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = user_interact_model.generate(batch=[promptRequest])

        cropped_respone_request_batch = []
        for res in respone:
            temp1 = res.split('\nRespone:', 1)[1]
            cropped_respone_request_batch.append(temp)

        print(f"Model respone:\n{cropped_respone_request_batch[0]}")

        input_text = input("Do you have any further modification (press 'done' if no, press 'yes' if yes): ") 

    json_object = json.dumps({"user_interact_result": cropped_respone_batch})
    with open("/content/user_interact_result.json", "w") as outfile:
        outfile.write(json_object)