from dataclasses import dataclass
from typing import Tuple, List, Dict, Set

import math
import mathutils
import numpy as np
from mathutils import Vector
import random
import bpy

@dataclass
class Layout:
    location: Tuple[float, float, float]
    min: Tuple[float, float, float]
    max: Tuple[float, float, float]
    orientation: Tuple[float, float, float] # Euler angles (pitch, yaw, roll)
    imported_object: bpy.types.Object # Object

def import_object(object_path: str, location: Tuple[float, float, float], orientation: Tuple[float, float, float]) -> bpy.types.Object:
    """
    Imports an .fbx object into Blender and positions/orients it according to the given parameters.

    Parameters:
    object_path (str): The path of the object to import (.fbx file) 
    location (Tuple[float, float, float]): The location (x, y, z) where the object should be placed.
    orientation (Tuple[float, float, float]): The orientation (pitch, yaw, roll in radians) of the object.

    Returns:
    bpy.types.Object: The imported object in Blender.
    """

    # Import .fbx file
    fbx_filepath = object_path  
    bpy.ops.import_scene.fbx(filepath=fbx_filepath)

    # Get a reference to the imported object
    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if imported_objects:
        imported_object = imported_objects[0]
        
        # Set location
        imported_object.location = location

        # Set orientation (convert Euler angles to radians)
        orientation_rad = tuple(angle for angle in orientation)
        imported_object.rotation_euler = orientation_rad

        # Example of calculating min and max bounds
        min_bound = (
            imported_object.location[0] - imported_object.dimensions[0] / 2,
            imported_object.location[1] - imported_object.dimensions[1] / 2,
            imported_object.location[2] - imported_object.dimensions[2] / 2
        )
        max_bound = (
            imported_object.location[0] + imported_object.dimensions[0] / 2,
            imported_object.location[1] + imported_object.dimensions[1] / 2,
            imported_object.location[2] + imported_object.dimensions[2] / 2
        )

        # Create and return Layout dataclass instance
        layout = Layout(location=imported_object.location,
                        min=min_bound,
                        max=max_bound,
                        orientation=orientation_rad,
                        imported_object=imported_object)
        
        return layout

    else:
        return None


def circle(x, y, z, step, number_step, speed, radius):
    angle = step/number_step * 2 * math.pi
    x = x + speed * radius * math.cos(angle)
    y = y + speed * radius * math.sin(angle)
    z = z
    return (x, y, z)

def straight(x, y, z, step, number_step, speed, orientation):
    x = x + speed * (number_step - step) * orientation[0] * 0.01
    y = y + speed * (number_step - step) * orientation[1] * 0.01
    z = z + speed * (number_step - step) * orientation[2] * 0.01
    return (x, y, z)

class Movement():
    def __init__(self, frame_start, frame_end, speed, trajectory, object):
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.speed = speed
        self.trajectory = trajectory 
        self.object = object

    def move(self, current_frame):
        if (current_frame >= self.frame_start) and (current_frame <= self.frame_end):    
            trajectory, param = self.trajectory

            # Current location
            x = self.object.imported_object.location[0]
            y = self.object.imported_object.location[1]
            z = self.object.imported_object.location[2]

            # The total number of step
            number_step = self.frame_end - self.frame_start + 1

            # Step
            step = current_frame - self.frame_start + 1 

            # Set the new location
            self.object.imported_object.location = trajectory(x=x, 
                                                              y=y, 
                                                              z=z, 
                                                              step=step,
                                                              number_step=number_step, 
                                                              speed=self.speed,
                                                              **param)

            # Insert keyframe for the new location
            self.object.imported_object.keyframe_insert(data_path="location", frame=current_frame)


if __name__ == '__main__':
    lion1 = import_object("/home/khai/Downloads/lion.fbx", location=(4, 7, 0), orientation=(0, 0, 0))
    lion2 = import_object("/home/khai/Downloads/lion.fbx", location=(5, 2, 0), orientation=(0, 0, 0))
    lion3 = import_object("/home/khai/Downloads/lion.fbx", location=(2, 3, 0), orientation=(0, 0, 0))

    num_frames = 200
    mov1 = Movement(frame_start=0, 
                    frame_end=100, 
                    trajectory=(circle, {'radius': 0.1}),
                    speed=1,
                    object=lion1)
    mov2 = Movement(frame_start=30, 
                    frame_end=90, 
                    trajectory=(circle, {'radius': 0.2}),
                    speed=1,
                    object=lion2)
    mov2 = Movement(frame_start=90, 
                    frame_end=150, 
                    trajectory=(straight, {'orientation': (1, 0, 1)}),
                    speed=1,
                    object=lion2)
    mov3 = Movement(frame_start=80, 
                    frame_end=150, 
                    trajectory=(circle, {'radius': 0.3}),
                    speed=2,
                    object=lion3)
    
    for frame in range(num_frames):
        mov1.move(current_frame=frame)
        mov2.move(current_frame=frame)
        mov3.move(current_frame=frame)

# Optionally, set the end frame of the animation
bpy.context.scene.frame_end = num_frames


