from RAG import RAG_module
from LanguageModel import VisionLangugeModel

if __name__ == '__main__':

# Kịch bản thử nghiệm 
# Đầu vào hệ thống gồm 3 phần:
#   - User Query: Chứa phần nội dung tương tác chính của người dùng tới mô hình ngôn ngữ
#   - External Text Resource: Chứa phần nội dung text thông tin người dùng muốn mô hình dựa vào để trả lời
#   - External Image Resource: Chứa hình ảnh, kèm discription, mà người dùng muốn thêm vào 
# -----------------------------------------------------------------------------
    # User Query
    input = 'Create 3D scene for this text: "A girl plays with her dog in the garden"'

    # External Text Resource
    external_text = """
The circular path that wraps around the lawn can be used for cycling around, running, and for the adults, walking around to admire the garden.
A sandpit or blowup plunge pool could be placed on the patio, a swing ball (if those still exist) or mini-trampoline on the lawn and suddenly the nicely designed adult garden is as welcoming for children as it is for adults without it having to look like a children's playground.
Dens can be made from cardboard boxes joined together, old chairs and tarpaulins and whatever else you've got stuffed away unused in the garage!
Make sure you put tough plants around lawn areas which can take having balls and small children land on them. So lavenders with their woody stems will fair much better than soft fleshy plants like hostas and herbaceous peonies.
"""

    # External Image Resource 
    images = ["Image1.jpeg",
              "Image2.jpeg",
              "Image3.jpeg"]
    
    images_discription = ["The garden should look like this",
                          "I want the girl character look like this",
                          "And the dog should look like this"] 
# -----------------------------------------------------------------------------

# Đối với External Text Resource, trựa tiếp đưa vào list các documents_augement
    documents_augement = [external_text]
    
# Đối với External Image Resource, mô hình ngôn ngữ chính sẽ đặt một số các câu hỏi liên quan đến các hình ảnh dựa vào: User query, Images_discription

    # Prompt để mô hình ngôn ngữ chính đặt câu hỏi
    for i, image_discription in enumerate(images_discription):
        question_query = f"""
You are an assistant that helps answer user requests.
The user provides you with some pictures, and you have to respond to their request based on those pictures.
However, you do not have direct access to the pictures. The only way to approach them is by asking questions related to the pictures to gather the necessary information.
Your task is to create questions based on the user's request and the description about the image to extract the needed information to fulfill the user's request.

Request: "{input}"

Description: "{image_discription}"

After getting the answers, format them as follows:
questions{i + 1} = [question1, question2, ...]

Avoid using normal text; format your response strictly as specified above.
"""

    question1s = """
questions1 = [
    "What are the main features of the garden in the picture? (e.g., flowers, trees, lawn, etc.)",
    "What type of dog is in the picture? (e.g., breed, size, color)",
    "What is the girl wearing in the picture? (e.g., dress, shorts, color, style)",
    "What is the overall color palette of the garden? (e.g., green, colorful, muted tones)",
    "Are there any notable objects or elements in the garden? (e.g., a bench, fountain, specific type of plant)",
    "What time of day does it appear to be in the picture? (e.g., morning, afternoon, evening)",
    "Is there any specific weather or lighting in the garden? (e.g., sunny, cloudy, shadowy)",
    "What is the general layout of the garden? (e.g., paths, flower beds, open space)",
    "Are there any additional characters or animals in the garden besides the girl and the dog?",
    "What is the background scenery beyond the garden? (e.g., fence, house, forest)"
]
"""

    question2s = """
questions2 = [
    "What is the girl wearing in the picture? (e.g., dress, shorts, color, style)",
    "What is the hairstyle of the girl in the picture? (e.g., long, short, braided, color)",
    "What is the girl's physical appearance in the picture? (e.g., height, build, facial features)",
    "What is the age or approximate age of the girl in the picture?",
    "Is the girl wearing any accessories in the picture? (e.g., hat, glasses, jewelry)",
    "What is the girl's pose or activity in the picture? (e.g., standing, sitting, running)",
    "What is the overall color palette or style of the girl's clothing in the picture? (e.g., bright, pastel, casual, formal)",
    "Are there any notable expressions or emotions visible on the girl's face in the picture?",
    "Does the girl have any distinguishing features in the picture? (e.g., freckles, scars, birthmarks)",
    "Is the girl holding or interacting with any objects in the picture? (e.g., toys, books, sports equipment)"
]
"""

    question3s = """
questions3 = [
    "What breed is the dog in the picture?",
    "What is the size of the dog in the picture? (e.g., small, medium, large)",
    "What is the color and pattern of the dog's fur in the picture?",
    "What is the dog's pose or activity in the picture? (e.g., sitting, running, playing)",
    "Does the dog have any accessories in the picture? (e.g., collar, leash, bandana)",
    "What is the dog's physical appearance in the picture? (e.g., ear shape, tail length, body build)",
    "Is there any notable expression or emotion visible on the dog's face in the picture?",
    "Are there any distinguishing features on the dog in the picture? (e.g., spots, scars, unique markings)",
    "What is the overall condition of the dog's fur in the picture? (e.g., well-groomed, messy, curly, straight)",
    "Is the dog interacting with any objects or characters in the picture? (e.g., toys, the girl, other animals)"
]
"""
    for i, _ in enumerate(images_discription):
        exec(f"exec(questions{i + 1})")

    # Các câu hỏi vừa xong tiếp đến sẽ được mô hình Vision_LM trả lời
    
    
    Vision_LM = VisionLangugeModel()

    for image in images:
        for question in questions:
            respone = Vision_LM.process(image_path=image, query=question)
            documents_augement.append(respone)
    

    # Initialize embedding database
    RAG_module = RAG_module
    for document in documents_augement:
        RAG_module.initalize_embedding_database(text=document)
    

    output_RAG = RAG_module.find_top_k_embedding(query=query, k=10)

