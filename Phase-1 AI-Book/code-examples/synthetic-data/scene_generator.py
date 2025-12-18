#!/usr/bin/env python3

"""
Synthetic Scene Generator
Generates 3D scenes with randomized objects for humanoid robotics training
"""

import cv2
import numpy as np
import math
import json
import os
from typing import List, Dict, Tuple, Any
import random
from dataclasses import dataclass


@dataclass
class Object3D:
    """Represents a 3D object in the scene"""
    name: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]  # Euler angles
    dimensions: Tuple[float, float, float]  # Width, height, depth
    color: Tuple[int, int, int]  # RGB
    object_type: str  # 'furniture', 'object', 'obstacle', etc.


class SceneGenerator:
    def __init__(self):
        # Define object categories and their properties
        self.object_categories = {
            "furniture": [
                {"name": "chair", "dims": (0.5, 0.8, 0.5), "color": (139, 69, 19)},
                {"name": "table", "dims": (1.0, 0.7, 0.6), "color": (160, 82, 45)},
                {"name": "sofa", "dims": (1.8, 0.8, 0.8), "color": (100, 100, 100)},
                {"name": "shelf", "dims": (0.8, 1.5, 0.3), "color": (139, 69, 19)}
            ],
            "objects": [
                {"name": "ball", "dims": (0.15, 0.15, 0.15), "color": (255, 0, 0)},
                {"name": "box", "dims": (0.2, 0.2, 0.2), "color": (0, 255, 0)},
                {"name": "bottle", "dims": (0.08, 0.25, 0.08), "color": (0, 0, 255)},
                {"name": "cup", "dims": (0.1, 0.1, 0.1), "color": (255, 255, 0)}
            ],
            "obstacles": [
                {"name": "plant", "dims": (0.3, 0.6, 0.3), "color": (0, 100, 0)},
                {"name": "lamp", "dims": (0.2, 1.2, 0.2), "color": (255, 255, 255)}
            ]
        }

        # Scene parameters
        self.scene_size = (5.0, 5.0)  # Width, depth of the room
        self.floor_height = 0.0
        self.ceiling_height = 2.5

        # Camera parameters (simulated)
        self.camera_height = 1.0  # Height of humanoid robot's camera
        self.camera_fov = 60  # Field of view in degrees

    def generate_random_object(self, obj_type: str) -> Object3D:
        """Generate a random object of specified type"""
        category_objects = self.object_categories.get(obj_type, [])
        if not category_objects:
            raise ValueError(f"Unknown object type: {obj_type}")

        # Select random object from category
        obj_template = random.choice(category_objects)

        # Generate random position within scene bounds
        x = random.uniform(0.5, self.scene_size[0] - 0.5)
        y = self.floor_height  # Objects on floor
        z = random.uniform(0.5, self.scene_size[1] - 0.5)

        # Generate random rotation
        rot_x = random.uniform(0, 2 * math.pi)
        rot_y = random.uniform(0, 2 * math.pi)
        rot_z = random.uniform(0, 2 * math.pi)

        # Add some randomization to dimensions
        dims = obj_template["dims"]
        dims = (
            dims[0] * random.uniform(0.8, 1.2),
            dims[1] * random.uniform(0.8, 1.2),
            dims[2] * random.uniform(0.8, 1.2)
        )

        return Object3D(
            name=f"{obj_template['name']}_{random.randint(1000, 9999)}",
            position=(x, y, z),
            rotation=(rot_x, rot_y, rot_z),
            dimensions=dims,
            color=obj_template["color"],
            object_type=obj_type
        )

    def generate_scene(self, num_objects: int = 10) -> List[Object3D]:
        """Generate a random scene with specified number of objects"""
        scene_objects = []

        # Define distribution of object types
        obj_type_probs = {
            "furniture": 0.3,
            "objects": 0.5,
            "obstacles": 0.2
        }

        for _ in range(num_objects):
            # Select object type based on probability
            rand_val = random.random()
            cumulative_prob = 0.0
            selected_type = "objects"  # default

            for obj_type, prob in obj_type_probs.items():
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_type = obj_type
                    break

            # Generate object
            obj = self.generate_random_object(selected_type)
            scene_objects.append(obj)

        return scene_objects

    def project_3d_to_2d(self, obj: Object3D, camera_pos: Tuple[float, float, float]) -> Tuple[int, int, int, int]:
        """Project 3D object to 2D bounding box (simplified projection)"""
        # Simplified orthographic projection for demonstration
        # In a real system, this would use perspective projection

        # Calculate relative position to camera
        rel_x = obj.position[0] - camera_pos[0]
        rel_z = obj.position[2] - camera_pos[2]

        # Calculate 2D position (simplified)
        center_x = int(320 + rel_x * 50)  # Scale factor for visualization
        center_y = int(240 - rel_z * 50)  # Invert Z for proper perspective

        # Calculate size based on distance (objects farther away appear smaller)
        distance = math.sqrt(rel_x**2 + rel_z**2)
        size_factor = max(0.1, 100 / (1 + distance))  # Size decreases with distance

        # Create bounding box
        width = int(obj.dimensions[0] * size_factor)
        height = int(obj.dimensions[1] * size_factor)

        x1 = center_x - width // 2
        y1 = center_y - height // 2
        x2 = center_x + width // 2
        y2 = center_y + height // 2

        return (x1, y1, x2, y2)

    def generate_synthetic_image(self, scene_objects: List[Object3D], width: int = 640, height: int = 480) -> np.ndarray:
        """Generate a synthetic RGB image of the scene"""
        # Create a base image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add a simple background (simulated walls/floor)
        image[:] = [100, 100, 100]  # Gray background

        # Add floor
        floor_level = int(height * 0.6)
        image[floor_level:, :] = [150, 120, 80]  # Brown floor

        # Add ceiling
        image[:int(height * 0.3), :] = [200, 200, 200]  # Light gray ceiling

        # Draw objects
        camera_pos = (self.scene_size[0] / 2, self.camera_height, self.scene_size[1] / 2)  # Center of room

        for obj in scene_objects:
            # Project 3D object to 2D
            bbox = self.project_3d_to_2d(obj, camera_pos)
            x1, y1, x2, y2 = bbox

            # Ensure bounding box is within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            if x2 > x1 and y2 > y1:
                # Draw the object as a colored rectangle
                color = obj.color
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # White border

                # Add object label
                cv2.putText(image, obj.name, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return image

    def generate_depth_map(self, scene_objects: List[Object3D], width: int = 640, height: int = 480) -> np.ndarray:
        """Generate a synthetic depth map"""
        depth_map = np.zeros((height, width), dtype=np.float32)
        depth_map.fill(10.0)  # Default depth (10 meters = far away)

        camera_pos = (self.scene_size[0] / 2, self.camera_height, self.scene_size[1] / 2)

        for obj in scene_objects:
            # Project to 2D
            bbox = self.project_3d_to_2d(obj, camera_pos)
            x1, y1, x2, y2 = bbox

            # Ensure bounding box is within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            if x2 > x1 and y2 > y1:
                # Calculate distance to object
                distance = math.sqrt(
                    (obj.position[0] - camera_pos[0])**2 +
                    (obj.position[1] - camera_pos[1])**2 +
                    (obj.position[2] - camera_pos[2])**2
                )

                # Fill the bounding box with depth value
                depth_map[y1:y2, x1:x2] = distance

        return depth_map

    def generate_annotations(self, scene_objects: List[Object3D], width: int = 640, height: int = 480) -> Dict[str, Any]:
        """Generate annotations for the scene"""
        camera_pos = (self.scene_size[0] / 2, self.camera_height, self.scene_size[1] / 2)

        annotations = {
            "scene_id": f"scene_{random.randint(10000, 99999)}",
            "image_size": [width, height],
            "camera_pose": camera_pos,
            "objects": []
        }

        for obj in scene_objects:
            bbox = self.project_3d_to_2d(obj, camera_pos)
            x1, y1, x2, y2 = bbox

            # Ensure bounding box is within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            # Calculate distance
            distance = math.sqrt(
                (obj.position[0] - camera_pos[0])**2 +
                (obj.position[1] - camera_pos[1])**2 +
                (obj.position[2] - camera_pos[2])**2
            )

            obj_annotation = {
                "name": obj.name,
                "type": obj.object_type,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "3d_position": obj.position,
                "3d_rotation": obj.rotation,
                "3d_dimensions": obj.dimensions,
                "color": obj.color,
                "distance": distance,
                "visibility": 1.0  # Fully visible in simulation
            }

            annotations["objects"].append(obj_annotation)

        return annotations

    def save_scene(self, scene_objects: List[Object3D], output_dir: str, scene_id: int):
        """Save the generated scene to disk"""
        os.makedirs(output_dir, exist_ok=True)

        # Generate synthetic data
        rgb_image = self.generate_synthetic_image(scene_objects)
        depth_map = self.generate_depth_map(scene_objects)
        annotations = self.generate_annotations(scene_objects)

        # Save RGB image
        cv2.imwrite(os.path.join(output_dir, f"rgb_{scene_id:05d}.png"), rgb_image)

        # Save depth map (convert to 16-bit for storage)
        depth_16bit = (depth_map * 1000).astype(np.uint16)  # Scale to mm
        cv2.imwrite(os.path.join(output_dir, f"depth_{scene_id:05d}.png"), depth_16bit)

        # Save annotations
        with open(os.path.join(output_dir, f"annotations_{scene_id:05d}.json"), 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"Saved scene {scene_id:05d} with {len(scene_objects)} objects")

    def generate_dataset(self, output_dir: str, num_scenes: int = 100):
        """Generate a complete synthetic dataset"""
        print(f"Generating {num_scenes} synthetic scenes...")

        for i in range(num_scenes):
            # Generate random number of objects for each scene
            num_objects = random.randint(5, 15)
            scene_objects = self.generate_scene(num_objects)

            # Save the scene
            self.save_scene(scene_objects, output_dir, i)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_scenes} scenes...")

        print(f"Dataset generation completed. Generated {num_scenes} scenes in {output_dir}")


def main():
    generator = SceneGenerator()

    print("Synthetic Data Generator for Humanoid Robotics")
    print("=" * 50)

    # Generate a single example scene to demonstrate
    print("Generating example scene...")
    example_scene = generator.generate_scene(num_objects=8)

    print(f"Generated scene with {len(example_scene)} objects:")
    for i, obj in enumerate(example_scene):
        print(f"  {i+1}. {obj.name} ({obj.object_type}) at {obj.position}")

    # Generate synthetic data for the example
    rgb_img = generator.generate_synthetic_image(example_scene)
    depth_map = generator.generate_depth_map(example_scene)
    annotations = generator.generate_annotations(example_scene)

    print(f"\nGenerated synthetic RGB image: {rgb_img.shape}")
    print(f"Generated depth map: {depth_map.shape}")
    print(f"Generated {len(annotations['objects'])} annotations")

    # Optionally generate a full dataset
    generate_full_dataset = False  # Set to True to generate a full dataset
    if generate_full_dataset:
        output_dir = "./synthetic_dataset"
        print(f"\nGenerating full dataset in {output_dir}...")
        generator.generate_dataset(output_dir, num_scenes=10)  # Generate 10 scenes for demo
    else:
        print("\nExample generation completed. Set generate_full_dataset=True to generate a full dataset.")

    print("\nSynthetic data generation system completed.")


if __name__ == '__main__':
    main()