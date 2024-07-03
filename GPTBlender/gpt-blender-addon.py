bl_info = {
    "name": "GPT Blender",
    "author": "Embedded Networking Laboratory",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "",
    "description": "Instant 3D content generation via LLM",
    "warning": "",
    "doc_url": "",
    "category": "3D content generation",
}

import bpy
from bpy.props import StringProperty, PointerProperty
from bpy.types import Panel, Operator, PropertyGroup
from bpy.utils import register_class, unregister_class

from g4f.client import Client



class GPTBlender_Properties(PropertyGroup):
    prompt: StringProperty(
        name="Prompt",
        description="User's prompt",
        default="",
        maxlen=4096,
    )
    
class GPTBlender_Operator(bpy.types.Operator):
    bl_idname = "genai.1"
    bl_label = "Generative AI Operator"
    
    def __init__(self):
        self.user_input = ""
        self.context = """
            Context: You're an assistant to help user with writing Python script in Blender.
            Your job is to generate a Blender-compatible Python snippet that is aligned 
            with user's intents. Your response should contain only Python code without any 
            further textual specifications. Ignore any query unrelated to Blender in which case
            you should return the message "Error".
            User's prompt:             
        """
        self.history = ""
        self.client = Client()
    
    def execute(self, context): 
        self.user_input = bpy.context.scene.GPTBlender_prop.prompt
        prompt = self.context + self.user_input
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        print(response.choices[0].message.content)
        

class GPTBlender_Panel(bpy.types.Panel):
    bl_label = "GPT Blender Panel"
    bl_idname = "OBJECT_PT_GPTBlender"
    # Define the area
    bl_space_type = "VIEW_3D"
    # Define the location of the panel
    bl_region_type = "UI"
    # Addon's name on the panel
    bl_category = "GPT Blender"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        GPTBlenderProp = scene.GPTBlender_prop
        # Text field
        layout.prop(GPTBlenderProp, "prompt")
        # Generate button
        row = layout.row()
        row.operator(GPTBlender_Operator.bl_idname, text="Generate", icon='SHADERFX')

_classes = [
    GPTBlender_Properties,
    GPTBlender_Operator,
    GPTBlender_Panel
]

def register():
    for cls in _classes:
        register_class(cls)
    bpy.types.Scene.GPTBlender_prop = PointerProperty(type=GPTBlender_Properties)
    
def unregister():
    for cls in _classes:
        unregister_class(cls)
    del bpy.types.Scene.GPTBlender_prop
    
if __name__ == "__main__":
    register()