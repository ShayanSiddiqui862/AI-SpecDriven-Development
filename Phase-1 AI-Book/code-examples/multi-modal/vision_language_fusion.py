#!/usr/bin/env python3

"""
Multi-Modal Vision-Language Fusion
Fuses visual and linguistic information for humanoid robot perception
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Any
import time
import random


class MultiModalFusion:
    def __init__(self):
        # Simulated CLIP-like model for vision-language understanding
        self.simulated_object_detections = {
            "red ball": {"confidence": 0.95, "bbox": [100, 100, 200, 200], "category": "object"},
            "blue chair": {"confidence": 0.89, "bbox": [300, 200, 450, 400], "category": "furniture"},
            "wooden table": {"confidence": 0.92, "bbox": [50, 300, 250, 450], "category": "furniture"},
            "person": {"confidence": 0.91, "bbox": [400, 150, 480, 350], "category": "person"}
        }

        # Simulated language understanding
        self.simulated_language_understanding = {
            "red ball": {"action": "grasp", "target": "red ball", "priority": 1},
            "blue chair": {"action": "navigate_to", "target": "blue chair", "priority": 2},
            "person": {"action": "greet", "target": "person", "priority": 3},
            "table": {"action": "avoid", "target": "table", "priority": 4}
        }

        # Attention weights for different modalities
        self.visual_weight = 0.7
        self.language_weight = 0.3

        # Scene understanding
        self.scene_objects = []
        self.language_instruction = ""
        self.attention_map = None

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Simulate object detection in the image"""
        print("Detecting objects in image...")

        # In a real implementation, this would use a real object detection model
        # For simulation, we'll return some pre-defined objects
        detected_objects = []
        for obj_name, obj_data in self.simulated_object_detections.items():
            # Add some randomness to detection
            if random.random() > 0.2:  # 80% chance of detection
                obj_copy = obj_data.copy()
                obj_copy["name"] = obj_name
                # Add some noise to bounding box
                bbox = obj_copy["bbox"]
                noise = random.uniform(-10, 10)
                obj_copy["bbox"] = [
                    max(0, bbox[0] + noise),
                    max(0, bbox[1] + noise),
                    min(image.shape[1], bbox[2] + noise),
                    min(image.shape[0], bbox[3] + noise)
                ]
                detected_objects.append(obj_copy)

        return detected_objects

    def understand_language(self, text: str) -> Dict[str, Any]:
        """Simulate language understanding"""
        print(f"Understanding language instruction: '{text}'")

        # Parse the text to identify objects and actions
        detected_objects = []
        for obj_name in self.simulated_language_understanding.keys():
            if obj_name.lower() in text.lower():
                if obj_name not in [obj["name"] for obj in detected_objects]:
                    obj_info = self.simulated_language_understanding[obj_name].copy()
                    obj_info["name"] = obj_name
                    detected_objects.append(obj_info)

        # If no specific objects mentioned, return general understanding
        if not detected_objects:
            return {
                "instruction": text,
                "action": "explore",
                "target": "environment",
                "priority": 5
            }

        return {
            "instruction": text,
            "objects": detected_objects,
            "primary_object": detected_objects[0] if detected_objects else None
        }

    def compute_visual_attention(self, image: np.ndarray, detected_objects: List[Dict]) -> np.ndarray:
        """Compute visual attention map based on detected objects"""
        height, width = image.shape[:2]
        attention_map = np.zeros((height, width), dtype=np.float32)

        for obj in detected_objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Create a Gaussian-like attention around the object
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2

            for y in range(y1, y2):
                for x in range(x1, x2):
                    distance = math.sqrt((x - obj_center_x)**2 + (y - obj_center_y)**2)
                    attention_value = max(0, 1 - distance / max(width, height) * 2)
                    attention_map[y, x] = max(attention_map[y, x], attention_value * obj["confidence"])

        return attention_map

    def compute_language_attention(self, language_info: Dict, image_shape: Tuple) -> np.ndarray:
        """Compute language-based attention map"""
        height, width = image_shape[:2]
        attention_map = np.zeros((height, width), dtype=np.float32)

        if "objects" in language_info:
            for obj in language_info["objects"]:
                # For simulation, create attention around the object's expected location
                # In reality, this would come from spatial language understanding
                if "bbox" in obj:  # If we have spatial information from vision
                    bbox = obj["bbox"]
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    attention_map[y1:y2, x1:x2] = 0.8
                else:
                    # If no spatial info, create a general attention map
                    attention_map[:, :] = 0.3

        return attention_map

    def fuse_modalities(self, visual_attention: np.ndarray, language_attention: np.ndarray) -> np.ndarray:
        """Fuse visual and language attention maps"""
        # Weighted fusion of attention maps
        fused_attention = (self.visual_weight * visual_attention +
                          self.language_weight * language_attention)

        # Normalize to [0, 1]
        if fused_attention.max() > 0:
            fused_attention = fused_attention / fused_attention.max()

        return fused_attention

    def process_instruction(self, image: np.ndarray, instruction: str) -> Dict[str, Any]:
        """Process a multi-modal instruction combining vision and language"""
        print(f"Processing multi-modal instruction: '{instruction}'")

        # Step 1: Detect objects in the image
        detected_objects = self.detect_objects(image)

        # Step 2: Understand the language instruction
        language_info = self.understand_language(instruction)

        # Step 3: Compute attention maps
        visual_attention = self.compute_visual_attention(image, detected_objects)
        language_attention = self.compute_language_attention(language_info, image.shape)
        fused_attention = self.fuse_modalities(visual_attention, language_attention)

        # Step 4: Identify the most relevant object based on fused attention
        most_relevant_object = None
        max_attention = 0

        for obj in detected_objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            # Calculate average attention in the object's region
            obj_attention = np.mean(fused_attention[y1:y2, x1:x2])
            if obj_attention > max_attention:
                max_attention = obj_attention
                most_relevant_object = obj

        # Step 5: Generate action based on fused understanding
        action = self.generate_action(most_relevant_object, language_info, instruction)

        result = {
            "instruction": instruction,
            "detected_objects": detected_objects,
            "language_analysis": language_info,
            "visual_attention": visual_attention,
            "language_attention": language_attention,
            "fused_attention": fused_attention,
            "most_relevant_object": most_relevant_object,
            "action": action,
            "confidence": max_attention if most_relevant_object else 0.0
        }

        return result

    def generate_action(self, relevant_object: Dict, language_info: Dict, instruction: str) -> Dict[str, Any]:
        """Generate appropriate action based on fused information"""
        if relevant_object is None:
            return {
                "action_type": "explore",
                "description": "No relevant objects detected, exploring environment",
                "parameters": {}
            }

        # Determine action based on language instruction and detected object
        obj_name = relevant_object["name"]
        action_type = "navigate_to"  # default action

        # Look for specific action in language understanding
        if "objects" in language_info:
            for lang_obj in language_info["objects"]:
                if lang_obj["name"] == obj_name:
                    action_type = lang_obj.get("action", action_type)
                    break

        # Generate action parameters
        action_params = {
            "target_object": obj_name,
            "bbox": relevant_object["bbox"],
            "confidence": relevant_object["confidence"]
        }

        # Special handling for different action types
        if action_type == "grasp":
            action_params["approach_direction"] = "top_down"
            action_params["gripper_width"] = 0.05  # 5cm
        elif action_type == "navigate_to":
            # Calculate center of object as navigation target
            bbox = relevant_object["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            action_params["target_position"] = [center_x, center_y]
        elif action_type == "greet":
            action_params["approach_distance"] = 1.0  # 1 meter

        return {
            "action_type": action_type,
            "description": f"{action_type.replace('_', ' ')} {obj_name}",
            "parameters": action_params
        }

    def visualize_result(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Visualize the multi-modal fusion result"""
        vis_image = image.copy()

        # Draw detected objects
        for obj in result["detected_objects"]:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            confidence = obj["confidence"]

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{obj['name']}: {confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Highlight most relevant object
        if result["most_relevant_object"]:
            obj = result["most_relevant_object"]
            bbox = obj["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Draw action
            action_text = f"Action: {result['action']['action_type']}"
            cv2.putText(vis_image, action_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display instruction
        cv2.putText(vis_image, f"Instruction: {result['instruction']}", (10, vis_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_image


def main():
    fusion = MultiModalFusion()

    print("Multi-Modal Vision-Language Fusion System")
    print("=" * 50)

    # Create a sample image (simulated environment)
    sample_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    # Add some colored rectangles to simulate objects
    cv2.rectangle(sample_image, (100, 100), (200, 200), (0, 0, 255), -1)  # Red ball
    cv2.rectangle(sample_image, (300, 200), (450, 400), (255, 0, 0), -1)  # Blue chair
    cv2.rectangle(sample_image, (50, 300), (250, 450), (139, 69, 19), -1)  # Table

    # Example instructions
    instructions = [
        "pick up the red ball",
        "go to the blue chair",
        "find the person",
        "avoid the table"
    ]

    for instruction in instructions:
        print(f"\nProcessing instruction: '{instruction}'")
        print("-" * 40)

        # Process the instruction
        result = fusion.process_instruction(sample_image, instruction)

        print(f"Detected {len(result['detected_objects'])} objects")
        if result['most_relevant_object']:
            obj = result['most_relevant_object']
            print(f"Most relevant object: {obj['name']} (confidence: {obj['confidence']:.2f})")
        print(f"Action: {result['action']['description']}")
        print(f"Action confidence: {result['confidence']:.2f}")

        # Visualize result (in a real system, this would show the image)
        vis_image = fusion.visualize_result(sample_image, result)
        print("Visualization created (simulated)")

    print("\nMulti-modal fusion system completed.")


if __name__ == '__main__':
    main()