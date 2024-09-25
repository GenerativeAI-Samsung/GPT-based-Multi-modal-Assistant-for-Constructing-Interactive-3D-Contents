from RAG import RAG_module
from LanguageModel import VisionLangugeModel, UserInteractModel, TestUserInteractModel
import json

if __name__ == '__main__':

# Kịch bản thử nghiệm 
# Đầu vào hệ thống gồm 3 phần:
#   - User Query: Chứa phần nội dung tương tác chính của người dùng tới mô hình ngôn ngữ
#   - External Text Resource: Chứa phần nội dung text thông tin người dùng muốn mô hình dựa vào để trả lời
#   - External Image Resource: Chứa hình ảnh, kèm discription, mà người dùng muốn thêm vào 
# -----------------------------------------------------------------------------
    # User Query
    input_text = input("Please provide your command: ")

    # External Text Resource
    external_texts = input("Please provide external text: ")

    external_images = []
    # External Image Resource 
    print("Please provide image path and description: ")
    control = None
    while (control != 'done'):
        temp_image_path = input("Please provide image_path: ")
        temp_image_description = input("Please provide image description: ")
        external_images.append({"image_path": temp_image_path, 
               "image_description": temp_image_description,
               "image_questions": None,
               "questions_response": None})
        control = input("Are you done yet? (press 'done' if done, else press 'no'): ")

# -----------------------------------------------------------------------------

    MODEL_ID = "LoftQ/Meta-Llama-3-8B-4bit-64rank"
# Khởi tạo Llama3-8B-Quantization
    print("------------------------------------------------------")
    print("Initialize Llama3-8B-Quantization...")
    # user_interact_model = UserInteractModel(MODEL_ID=MODEL_ID)
    user_interact_model = TestUserInteractModel()
    print("Done!")
    print("------------------------------------------------------")

# Đối với External Text Resource, trựa tiếp đưa vào list các documents_augement
    documents_augement = [external_texts]
    
# Đối với External Image Resource, mô hình ngôn ngữ chính sẽ đặt một số các câu hỏi liên quan đến các hình ảnh dựa vào: User query, Images_discription
    print("\n------------------------------------------------------")
    print('User Interact Model is asking questions about image...')
    # Prompt để mô hình ngôn ngữ chính đặt câu hỏi
    question_asked = []
    for i, image in enumerate(external_images):
        prompt = f"""
You are an assistant that helps answer user requests.
The user provides you with some pictures, and you have to respond to their request based on those pictures.
However, you do not have direct access to the pictures. The only way to approach them is by asking questions related to the pictures to gather the necessary information.
Your task is to create questions based on the user's request and the description about the image to extract the needed information to fulfill the user's request.

Request: "{input_text}"

Description: "{image["image_description"]}"

After getting the answers, format them as follows:
external_images[{i}]["image_questions"] = [question1, question2, ...]

Avoid using normal text; format your response strictly as specified above.
"""
        respone = user_interact_model.generate(batch=[prompt])
        question_asked.append({"index": i, "prompt": prompt, "respone": respone[0]})
    print("Done!")
    print("------------------------------------------------------")

    print("\n------------------------------------------------------")
    # Viết ra file .json và yêu cầu người dùng check lại
    json_object = json.dumps(question_asked, indent=4)
    with open("question_asked.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in question_asked.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    # Đọc lại file .json để đến bước tiếp theo
    with open('question_asked.json', 'r') as openfile:
        question_asked = json.load(openfile)
    print("------------------------------------------------------")

    # Thực thi các câu hỏi đó
    for item in question_asked:
        exec(item["respone"])

    # Các câu hỏi vừa xong tiếp đến sẽ được mô hình Vision_LM trả lời
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
            # Prompt cho mô hình Vision Language trả lời, dựa vào: Hình ảnh, Image description, Image Question
            prompt = f"""
Based on the image and the description of the picture below, please answer the following questions related to the image
Description of the image: {image["image_description"]}
Question: {question}
"""
            respone = Vision_LM.process(image_path=image["image_path"],query=prompt)
            temp_list.append({"image_index": i, "question": question, "respone": respone})
        answers.append(temp_list)
    
    print("\n------------------------------------------------------")
    # Viết ra file .json và yêu cầu người dùng check lại
    json_object = json.dumps(answers, indent=4)
    with open("question_asked.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in answers.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    # Đọc lại file .json để đến bước tiếp theo
    with open('answers.json', 'r') as openfile:
        answers = json.load(openfile)
    print("------------------------------------------------------")

    # Cập nhập câu trả lời vào external_images
    for image in external_images:
        image["questions_response"] = [item["respone"] for item in answers]

    # Các câu hỏi và câu trả lời của mô hình ngôn ngữ sẽ được đưa vào list các documents_augement để về sau mô hình ngôn ngữ truy vấn
    for image in external_images:
        documents_augement.extend(f"question: {question}, respone: {respone}"  for respone, question in zip(image["questions_response"], image["image_questions"]))

    # Khởi tạo RAG module
    print("\n------------------------------------------------------")
    print("Initialize RAG Module") 
    Retrieval_module = RAG_module()
    Retrieval_module.initalize_embedding_database(text=documents_augement)

    print("------------------------------------------------------")

    print("Start User Interact Interface...")
    while (input_text != 'done'):
        # Thực hiện việc truy xuất thông tin từ RAG module (Top 20)
        output_RAG = RAG_module.find_top_k_embedding(query=input_text, k=20)

        # Sau đó, prompt để mô hình ngôn ngữ trả lời dựa trên thông tin được lựa từ output_RAG
        prompt = f"""
You are a friendly assistant. Your task is to interact with the user to create a script that meets the user's requirements.

User input: {input_text} 

Additionally, there is some supplementary information that will help you respond more accurately to the user's needs:
{''.join((item + "\n") for item in output_RAG)}

Your answer should contain natural language only
    """
        print("responing...")
        respone = user_interact_model.generate(batch=[prompt])
        print(f"Model respone:\n{respone[0]}")

        Retrieval_module.initalize_embedding_database(text=respone[0])
        input_text = input("Do you have any further modification (press 'done' if no, press 'yes' if yes): ") 

    # lưu lại các embedding vector thành file .json
    json_object = json.dumps(Retrieval_module.embedding_dicts)
    with open("/content/embedding_dicts.json", "w") as outfile:
        outfile.write(json_object)
    
    # Lưu câu trả lời lại để sử dụng cho phần sau
    json_object = json.dumps(respone)
    with open("/content/user_interact_result.json", "w") as outfile:
        outfile.write(json_object)