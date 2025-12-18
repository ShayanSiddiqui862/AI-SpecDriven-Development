---
sidebar_position: 42
---

# Multi-Modal Fusion for Vision-Language-Action Systems

## Learning Objectives
By the end of this module, students will be able to:
- Implement multi-modal fusion architectures for combining vision, language, and action modalities
- Design attention mechanisms for cross-modal alignment
- Integrate sensor data fusion with language understanding for robotics applications
- Create vision-language models for robotic manipulation tasks
- Evaluate fusion performance and optimize for real-time applications

## Theory

### Multi-Modal Fusion in Robotics

Multi-modal fusion is the process of combining information from different sensory modalities (vision, language, tactile, etc.) to create a more comprehensive understanding of the environment and enable better decision-making. In robotics, this is particularly important for:

- **Perception**: Combining visual and linguistic cues for object recognition
- **Planning**: Using language commands to guide visual-based path planning
- **Action**: Executing actions based on combined visual and linguistic input

### Key Concepts

#### 1. Modalities in Robotics
- **Vision**: Camera feeds, LiDAR point clouds, depth information
- **Language**: Natural language commands, descriptions, queries
- **Action**: Robot motor commands, proprioceptive feedback
- **Tactile**: Force/torque sensors, contact information
- **Audio**: Sound, speech, environmental audio

#### 2. Fusion Strategies
- **Early Fusion**: Combine raw data from different modalities early in the pipeline
- **Late Fusion**: Process modalities separately and combine high-level features
- **Intermediate Fusion**: Combine features at intermediate processing layers
- **Attention-based Fusion**: Use attention mechanisms to weigh different modalities

#### 3. Cross-Modal Alignment
- **Grounding**: Associating linguistic concepts with visual entities
- **Captioning**: Generating language descriptions of visual scenes
- **Referring Expression**: Understanding language referring to specific visual objects
- **Visual Question Answering**: Answering questions about visual content

### Vision-Language-Action (VLA) Models

#### 1. Architecture Components
- **Visual Encoder**: Processes images/videos to extract visual features
- **Language Encoder**: Processes text to extract linguistic features
- **Fusion Module**: Combines visual and language features
- **Action Decoder**: Generates action sequences from fused features

#### 2. Attention Mechanisms
- **Self-Attention**: Within-modal attention for refining modality-specific features
- **Cross-Attention**: Between-modal attention for cross-modal alignment
- **Multi-Head Attention**: Multiple attention heads for capturing different relationships

## Implementation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- Transformers library
- OpenCV
- ROS 2 Humble with vision packages

### 1. Basic Multi-Modal Fusion Architecture

#### Vision-Language Fusion Module
```python
# multimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
import clip
from typing import Dict, List, Tuple, Optional
import numpy as np


class VisionLanguageFusion(nn.Module):
    """
    Basic vision-language fusion module for robotics applications.
    """

    def __init__(self, visual_dim: int = 512, text_dim: int = 512, fusion_dim: int = 1024):
        super().__init__()

        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim

        # Visual encoder (using CLIP vision encoder as example)
        self.visual_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        # Text encoder (using CLIP text encoder as example)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        # Cross-attention fusion module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1
        )

        # Fusion layers
        self.visual_projection = nn.Linear(visual_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )

        # Output head for action prediction
        self.action_head = nn.Linear(fusion_dim, 256)  # Adjust based on action space

    def forward(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Forward pass for vision-language fusion.

        Args:
            images: Batch of images [B, C, H, W]
            texts: List of text descriptions [B]

        Returns:
            Fused features [B, fusion_dim]
        """
        # Encode visual features
        visual_features = self.encode_visual(images)  # [B, seq_len, visual_dim]

        # Encode text features
        text_features = self.encode_text(texts)  # [B, seq_len, text_dim]

        # Project to fusion dimension
        visual_proj = self.visual_projection(visual_features)  # [B, seq_len, fusion_dim]
        text_proj = self.text_projection(text_features)  # [B, seq_len, fusion_dim]

        # Cross-attention fusion
        fused_features = self.cross_attention_fusion(visual_proj, text_proj)

        # Apply fusion layer
        output = self.fusion_layer(fused_features)

        return output

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual features using vision encoder."""
        with torch.no_grad():
            visual_outputs = self.visual_encoder(images)
        return visual_outputs.last_hidden_state  # [B, seq_len, dim]

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text features using text encoder."""
        # Tokenize texts (in real implementation, use proper tokenizer)
        # For now, we'll simulate text encoding
        batch_size = len(texts)
        seq_len = 10  # Example sequence length
        text_features = torch.randn(batch_size, seq_len, self.text_dim)
        return text_features

    def cross_attention_fusion(self, visual_features: torch.Tensor,
                              text_features: torch.Tensor) -> torch.Tensor:
        """Fuse visual and text features using cross-attention."""
        # Concatenate visual and text features
        combined_features = torch.cat([visual_features, text_features], dim=1)  # [B, 2*seq_len, fusion_dim]

        # Apply cross-attention
        attended_features, attention_weights = self.cross_attention(
            query=combined_features,
            key=combined_features,
            value=combined_features
        )

        # Global average pooling to get fixed-size representation
        fused_features = attended_features.mean(dim=1)  # [B, fusion_dim]

        return fused_features


class VisionLanguageActionFusion(nn.Module):
    """
    Extended VLA fusion module that includes action prediction.
    """

    def __init__(self, visual_dim: int = 512, text_dim: int = 512,
                 fusion_dim: int = 1024, action_space: int = 10):
        super().__init__()

        self.vision_language_fusion = VisionLanguageFusion(visual_dim, text_dim, fusion_dim)
        self.action_predictor = nn.Linear(fusion_dim, action_space)
        self.action_space = action_space

    def forward(self, images: torch.Tensor, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VLA fusion.

        Args:
            images: Batch of images [B, C, H, W]
            texts: List of text descriptions [B]

        Returns:
            Dictionary with fused features and action predictions
        """
        # Get fused features
        fused_features = self.vision_language_fusion(images, texts)

        # Predict actions
        action_logits = self.action_predictor(fused_features)
        action_probs = F.softmax(action_logits, dim=-1)

        return {
            'fused_features': fused_features,
            'action_logits': action_logits,
            'action_probs': action_probs
        }


# Example usage
if __name__ == "__main__":
    # Initialize fusion model
    vla_fusion = VisionLanguageActionFusion(
        visual_dim=768,  # CLIP vision feature dimension
        text_dim=512,    # CLIP text feature dimension
        fusion_dim=1024,
        action_space=20  # Example action space
    )

    # Example inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)  # Example images
    texts = ["Pick up the red cup", "Move to the kitchen", "Open the door", "Sit on the chair"]

    # Forward pass
    outputs = vla_fusion(images, texts)

    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Action logits shape: {outputs['action_logits'].shape}")
    print(f"Action probabilities shape: {outputs['action_probs'].shape}")
```

#### 2. Advanced Fusion with Spatial Attention
```python
# spatial_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel, BertTokenizer


class SpatialVisionLanguageFusion(nn.Module):
    """
    Advanced fusion module with spatial attention for precise grounding.
    """

    def __init__(self, visual_backbone: str = 'resnet50', hidden_dim: int = 512):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Visual backbone
        self.visual_backbone = getattr(models, visual_backbone)(pretrained=True)
        # Remove classifier
        self.visual_backbone = nn.Sequential(*list(self.visual_backbone.children())[:-2])

        # Visual feature processing
        self.visual_conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)  # Adjust based on backbone output

        # Text encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Spatial attention module
        self.spatial_attention = SpatialAttentionModule(hidden_dim)

        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(hidden_dim)

        # Action decoder
        self.action_decoder = ActionDecoder(hidden_dim, action_space=256)

    def forward(self, images: torch.Tensor, text_queries: List[str]) -> Dict:
        """
        Forward pass with spatial grounding.

        Args:
            images: Batch of images [B, C, H, W]
            text_queries: List of text queries [B]

        Returns:
            Dictionary with spatial attention maps and action predictions
        """
        # Extract visual features (spatial features)
        visual_features = self.visual_backbone(images)  # [B, C, H', W']
        visual_features = self.visual_conv(visual_features)  # [B, hidden_dim, H', W']

        # Extract text features
        text_features, attention_mask = self.encode_text(text_queries)

        # Apply spatial attention for grounding
        spatial_attended_features = self.spatial_attention(visual_features, text_features)

        # Cross-modal attention
        fused_features = self.cross_modal_attention(
            spatial_attended_features, text_features, attention_mask
        )

        # Decode actions
        actions = self.action_decoder(fused_features)

        return {
            'spatial_attention_maps': spatial_attended_features,
            'fused_features': fused_features,
            'actions': actions,
            'text_features': text_features
        }

    def encode_text(self, text_queries: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text queries using BERT."""
        encoded = self.tokenizer(
            text_queries,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=64
        )

        with torch.no_grad():
            text_outputs = self.text_encoder(**encoded)

        # Use [CLS] token representation
        text_features = text_outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        attention_mask = encoded.attention_mask  # [B, seq_len]

        return text_features, attention_mask


class SpatialAttentionModule(nn.Module):
    """
    Module for spatial attention in vision-language fusion.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Text-guided spatial attention
        self.text_guided_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

        # Spatial feature processing
        self.spatial_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, visual_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention guided by text features.

        Args:
            visual_features: [B, hidden_dim, H, W]
            text_features: [B, seq_len, hidden_dim]

        Returns:
            Spatially attended visual features [B, hidden_dim, H, W]
        """
        batch_size, _, height, width = visual_features.shape

        # Average text features across sequence dimension
        text_avg = text_features.mean(dim=1)  # [B, hidden_dim]

        # Generate spatial attention weights from text guidance
        text_attention_weights = self.text_guided_attention(text_avg)  # [B, hidden_dim]

        # Reshape to match spatial dimensions
        text_attention_weights = text_attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, hidden_dim, 1, 1]

        # Apply attention to visual features
        attended_visual = visual_features * text_attention_weights  # [B, hidden_dim, H, W]

        # Apply spatial convolution
        attended_visual = self.spatial_conv(attended_visual)

        return attended_visual


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between visual and text features.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, visual_features: torch.Tensor,
                text_features: torch.Tensor,
                text_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Cross-modal attention between visual and text features.

        Args:
            visual_features: [B, hidden_dim, H, W] - flattened to [B, spatial_dim, hidden_dim]
            text_features: [B, seq_len, hidden_dim]
            text_attention_mask: [B, seq_len]

        Returns:
            Cross-attended features [B, spatial_dim, hidden_dim]
        """
        batch_size, hidden_dim, height, width = visual_features.shape

        # Flatten spatial dimensions
        visual_flat = visual_features.view(batch_size, hidden_dim, -1).transpose(1, 2)  # [B, spatial_dim, hidden_dim]

        # Concatenate visual and text features
        all_features = torch.cat([visual_flat, text_features], dim=1)  # [B, spatial_dim + seq_len, hidden_dim]

        # Apply projections
        Q = self.query_proj(all_features)
        K = self.key_proj(all_features)
        V = self.value_proj(all_features)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, total_seq, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, total_seq, total_seq]

        # Apply attention mask (extend for visual features)
        extended_mask = torch.ones(batch_size, visual_flat.size(1), dtype=torch.bool, device=text_attention_mask.device)  # [B, spatial_dim]
        combined_mask = torch.cat([extended_mask, text_attention_mask.bool()], dim=1)  # [B, total_seq]
        combined_mask = combined_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, total_seq]

        # Mask attention scores
        attention_scores = attention_scores.masked_fill(combined_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, V)  # [B, num_heads, total_seq, head_dim]

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, hidden_dim)  # [B, total_seq, hidden_dim]

        # Project output
        output = self.out_proj(attended)

        # Return only visual portion (first spatial_dim elements)
        return output[:, :visual_flat.size(1), :]  # [B, spatial_dim, hidden_dim]


class ActionDecoder(nn.Module):
    """
    Action decoder for generating robot actions from fused features.
    """

    def __init__(self, hidden_dim: int, action_space: int, max_action_length: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.max_action_length = max_action_length

        # Action sequence decoder
        self.action_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Action prediction head
        self.action_predictor = nn.Linear(hidden_dim, action_space)

        # Sequence length predictor
        self.length_predictor = nn.Linear(hidden_dim, max_action_length)

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode actions from fused features.

        Args:
            fused_features: [B, spatial_dim, hidden_dim]

        Returns:
            Dictionary with action sequences and lengths
        """
        batch_size = fused_features.size(0)

        # Use global pooled features as initial state
        global_features = fused_features.mean(dim=1)  # [B, hidden_dim]

        # Predict action sequence length
        length_logits = self.length_predictor(global_features)  # [B, max_action_length]
        predicted_lengths = F.softmax(length_logits, dim=-1)

        # Repeat global features for sequence generation
        repeated_features = global_features.unsqueeze(1).repeat(1, self.max_action_length, 1)  # [B, max_len, hidden_dim]

        # Generate action sequence
        lstm_out, _ = self.action_lstm(repeated_features)  # [B, max_len, hidden_dim]

        # Predict actions
        action_logits = self.action_predictor(lstm_out)  # [B, max_len, action_space]

        return {
            'action_sequences': action_logits,
            'predicted_lengths': predicted_lengths,
            'action_probs': F.softmax(action_logits, dim=-1)
        }
```

### 3. ROS 2 Integration for Multi-Modal Fusion

#### ROS 2 Node for VLA Processing
```python
#!/usr/bin/env python3
"""
ROS 2 node for multi-modal fusion of vision, language, and action.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from robotics_demo_msgs.msg import ActionSequence, RobotAction
from cv_bridge import CvBridge
import torch
import numpy as np
from multimodal_fusion import VisionLanguageActionFusion
import json


class MultiModalFusionNode(Node):
    """
    ROS 2 node that performs multi-modal fusion of vision, language, and action.
    """

    def __init__(self):
        super().__init__('multi_modal_fusion_node')

        # Parameters
        self.declare_parameter('fusion_model_path', '/path/to/fusion/model.pth')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('max_action_length', 10)

        self.model_path = self.get_parameter('fusion_model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.max_action_length = self.get_parameter('max_action_length').value

        # Initialize fusion model
        self.get_logger().info('Loading multi-modal fusion model...')
        self.fusion_model = VisionLanguageActionFusion(
            visual_dim=768,
            text_dim=512,
            fusion_dim=1024,
            action_space=256
        )

        # Load trained weights if available
        try:
            checkpoint = torch.load(self.model_path)
            self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
            self.get_logger().info('Fusion model loaded successfully')
        except FileNotFoundError:
            self.get_logger().warn(f'Model file not found at {self.model_path}, using random initialization')
        except Exception as e:
            self.get_logger().error(f'Error loading model: {e}')

        # Set model to evaluation mode
        self.fusion_model.eval()

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers
        self.action_pub = self.create_publisher(ActionSequence, '/robot/action_sequence', 10)
        self.attention_pub = self.create_publisher(Image, '/fusion/attention_map', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/voice_commands',
            self.command_callback,
            10
        )

        # Internal state
        self.latest_image = None
        self.pending_command = None
        self.fusion_lock = threading.Lock()

        self.get_logger().info('Multi-modal fusion node initialized')

    def image_callback(self, msg: Image):
        """Process incoming image data."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert to tensor (resize if necessary)
            image_tensor = self.preprocess_image(cv_image)

            with self.fusion_lock:
                self.latest_image = image_tensor

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg: String):
        """Process incoming language command."""
        command = msg.data

        with self.fusion_lock:
            self.pending_command = command

        # Trigger fusion if we have both image and command
        if self.latest_image is not None and self.pending_command is not None:
            self.perform_multimodal_fusion()

    def preprocess_image(self, cv_image: np.ndarray) -> torch.Tensor:
        """Preprocess image for fusion model."""
        # Resize image to expected input size
        resized = cv2.resize(cv_image, (224, 224))

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(resized).float().permute(2, 0, 1)  # [C, H, W]
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]

        # Normalize with ImageNet statistics
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - imagenet_mean) / imagenet_std

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]

        return image_tensor

    def perform_multimodal_fusion(self):
        """Perform fusion of vision and language modalities."""
        if self.latest_image is None or self.pending_command is None:
            return

        try:
            # Prepare inputs
            images = self.latest_image
            texts = [self.pending_command]

            # Perform fusion
            with torch.no_grad():
                outputs = self.fusion_model(images, texts)

            # Extract results
            action_probs = outputs['action_probs']
            fused_features = outputs['fused_features']

            # Apply confidence threshold
            max_probs, predicted_actions = torch.max(action_probs, dim=-1)

            # Filter by confidence
            confident_actions = []
            for i, (action, prob) in enumerate(zip(predicted_actions[0], max_probs[0])):
                if prob.item() > self.confidence_threshold:
                    confident_actions.append(action.item())

            if confident_actions:
                # Create action sequence message
                action_seq_msg = ActionSequence()
                action_seq_msg.header.stamp = self.get_clock().now().to_msg()
                action_seq_msg.header.frame_id = 'base_link'

                for action_id in confident_actions:
                    robot_action = RobotAction()
                    robot_action.action_id = action_id
                    robot_action.confidence = float(max_probs[0][confident_actions.index(action_id)].item())
                    robot_action.description = self.action_id_to_description(action_id)

                    action_seq_msg.actions.append(robot_action)

                # Publish action sequence
                self.action_pub.publish(action_seq_msg)

                self.get_logger().info(
                    f'Published action sequence: {[a.action_id for a in action_seq_msg.actions]}'
                )

            # Clear processed command
            self.pending_command = None

        except Exception as e:
            self.get_logger().error(f'Error in multi-modal fusion: {e}')

    def action_id_to_description(self, action_id: int) -> str:
        """Convert action ID to human-readable description."""
        # Define action mapping (this would be more comprehensive in practice)
        action_map = {
            0: "move_forward",
            1: "move_backward",
            2: "turn_left",
            3: "turn_right",
            4: "grasp_object",
            5: "release_object",
            6: "navigate_to",
            7: "detect_object",
            8: "speak",
            9: "stop"
        }

        return action_map.get(action_id, f"action_{action_id}")

    def create_attention_visualization(self, attention_weights: torch.Tensor) -> Image:
        """Create attention visualization for debugging."""
        # Convert attention weights to heatmap
        attention_map = attention_weights.squeeze().detach().cpu().numpy()

        # Normalize to [0, 255]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        attention_map = (attention_map * 255).astype(np.uint8)

        # Convert to color heatmap
        heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

        # Convert to ROS image
        attention_msg = self.cv_bridge.cv2_to_imgmsg(heatmap, "bgr8")
        attention_msg.header.stamp = self.get_clock().now().to_msg()
        attention_msg.header.frame_id = 'base_link'

        return attention_msg


def main(args=None):
    rclpy.init(args=args)

    fusion_node = MultiModalFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        fusion_node.get_logger().info('Shutting down multi-modal fusion node...')
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 4. Sensor Data Fusion

#### Multi-Sensor Fusion Pipeline
```python
# sensor_fusion.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs


class MultiSensorFusion(nn.Module):
    """
    Multi-sensor fusion for robotics applications combining vision, LiDAR, IMU, etc.
    """

    def __init__(self,
                 visual_dim: int = 512,
                 lidar_dim: int = 256,
                 imu_dim: int = 128,
                 fusion_dim: int = 1024):
        super().__init__()

        self.visual_dim = visual_dim
        self.lidar_dim = lidar_dim
        self.imu_dim = imu_dim
        self.fusion_dim = fusion_dim

        # Individual modality encoders
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, fusion_dim // 2)
        )

        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, fusion_dim // 2)
        )

        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 4, fusion_dim // 4)
        )

        # Cross-modal attention for fusion
        self.cross_attention = CrossModalAttention(fusion_dim)

        # Fusion module
        self.fusion_module = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )

        # Uncertainty estimation
        self.uncertainty_estimator = nn.Linear(fusion_dim, 1)

    def forward(self,
                visual_features: Optional[torch.Tensor] = None,
                lidar_features: Optional[torch.Tensor] = None,
                imu_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Fuse multiple sensor modalities.

        Args:
            visual_features: [B, visual_dim] or None
            lidar_features: [B, lidar_dim] or None
            imu_features: [B, imu_dim] or None

        Returns:
            Dictionary with fused features and uncertainty estimates
        """
        batch_size = self.get_batch_size(visual_features, lidar_features, imu_features)

        # Encode individual modalities
        encoded_features = []

        if visual_features is not None:
            visual_encoded = self.visual_encoder(visual_features)
            encoded_features.append(visual_encoded)

        if lidar_features is not None:
            lidar_encoded = self.lidar_encoder(lidar_features)
            encoded_features.append(lidar_encoded)

        if imu_features is not None:
            imu_encoded = self.imu_encoder(imu_features)
            encoded_features.append(imu_encoded)

        if not encoded_features:
            raise ValueError("At least one modality must be provided")

        # Concatenate encoded features
        if len(encoded_features) == 1:
            fused_features = encoded_features[0]
        else:
            # Use cross-attention to fuse features
            combined_features = torch.cat(encoded_features, dim=1)  # [B, total_dim]

            # Project to fusion dimension
            if combined_features.size(1) != self.fusion_dim:
                combined_features = nn.Linear(combined_features.size(1), self.fusion_dim)(combined_features)

            # Apply fusion
            fused_features = self.fusion_module(combined_features)

        # Estimate uncertainty
        uncertainty = torch.sigmoid(self.uncertainty_estimator(fused_features))

        return {
            'fused_features': fused_features,
            'uncertainty': uncertainty,
            'modalities_present': [f is not None for f in [visual_features, lidar_features, imu_features]]
        }

    def get_batch_size(self, *features) -> int:
        """Get batch size from the first non-None tensor."""
        for feat in features:
            if feat is not None:
                return feat.size(0)
        raise ValueError("All features are None")


class SensorFusionNode:
    """
    ROS 2 node for multi-sensor fusion.
    """

    def __init__(self):
        # Initialize fusion model
        self.fusion_model = MultiSensorFusion()

        # Publishers and subscribers
        self.fused_data_pub = None  # Would be initialized in ROS node

        # Sensor data buffers
        self.visual_buffer = []
        self.lidar_buffer = []
        self.imu_buffer = []

        # Timestamp alignment
        self.sync_window = 0.1  # 100ms window for synchronization

    def process_visual_data(self, visual_msg):
        """Process visual data from camera."""
        # Convert ROS image to features
        features = self.extract_visual_features(visual_msg)
        self.visual_buffer.append((visual_msg.header.stamp, features))

    def process_lidar_data(self, lidar_msg):
        """Process LiDAR data."""
        # Convert point cloud to features
        features = self.extract_lidar_features(lidar_msg)
        self.lidar_buffer.append((lidar_msg.header.stamp, features))

    def process_imu_data(self, imu_msg):
        """Process IMU data."""
        # Convert IMU data to features
        features = self.extract_imu_features(imu_msg)
        self.imu_buffer.append((imu_msg.header.stamp, features))

    def extract_visual_features(self, image_msg):
        """Extract visual features from image message."""
        # This would typically use a CNN to extract features
        # For now, returning dummy features
        return torch.randn(1, 512)

    def extract_lidar_features(self, pointcloud_msg):
        """Extract features from point cloud."""
        # This would typically process the point cloud
        # For now, returning dummy features
        return torch.randn(1, 256)

    def extract_imu_features(self, imu_msg):
        """Extract features from IMU data."""
        # Extract relevant IMU features
        lin_acc = torch.tensor([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ]).float().unsqueeze(0)

        return lin_acc

    def synchronize_and_fuse(self):
        """Synchronize sensor data and perform fusion."""
        # Find aligned timestamps
        aligned_data = self.find_aligned_timestamps()

        if aligned_data:
            visual_feat, lidar_feat, imu_feat = aligned_data

            # Perform fusion
            fusion_result = self.fusion_model(visual_feat, lidar_feat, imu_feat)

            # Publish fused data
            self.publish_fused_data(fusion_result)

    def find_aligned_timestamps(self):
        """Find timestamps that align across sensors within sync window."""
        # Find closest timestamps within the synchronization window
        # This is a simplified implementation
        if (self.visual_buffer and self.lidar_buffer and self.imu_buffer):
            # Take the most recent from each buffer
            visual_data = self.visual_buffer[-1][1] if self.visual_buffer else None
            lidar_data = self.lidar_buffer[-1][1] if self.lidar_buffer else None
            imu_data = self.imu_buffer[-1][1] if self.imu_buffer else None

            return visual_data, lidar_data, imu_data

        return None

    def publish_fused_data(self, fusion_result):
        """Publish fused sensor data."""
        # This would create and publish a ROS message with fused data
        fused_features = fusion_result['fused_features']
        uncertainty = fusion_result['uncertainty']

        print(f"Fused features shape: {fused_features.shape}")
        print(f"Uncertainty: {uncertainty.mean().item()}")


# Example usage
if __name__ == "__main__":
    fusion = MultiSensorFusion()

    # Example with all modalities
    visual_in = torch.randn(2, 512)
    lidar_in = torch.randn(2, 256)
    imu_in = torch.randn(2, 128)

    result = fusion(visual_in, lidar_in, imu_in)

    print(f"Fused features: {result['fused_features'].shape}")
    print(f"Uncertainty: {result['uncertainty'].shape}")
    print(f"Modalities present: {result['modalities_present']}")
```

### 5. Vision-Language Grounding

#### Grounding Module
```python
# vision_language_grounding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from typing import List, Tuple, Dict


class VisionLanguageGrounding(nn.Module):
    """
    Vision-language grounding module for connecting language descriptions to visual entities.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()

        # Load pre-trained CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Object detection backbone (could be DETR, YOLO, etc.)
        self.detection_backbone = self.load_detection_model()

        # Grounding head
        self.grounding_head = nn.Linear(512, 256)  # CLIP feature dim to grounding dim

        # Spatial localization head
        self.localization_head = nn.Linear(512, 4)  # [x, y, width, height]

    def load_detection_model(self):
        """
        Load object detection model (using DETR as example).
        In practice, you'd use a pre-trained detection model.
        """
        # This is a placeholder - would load actual detection model
        # For example: detr_resnet_50 from torchvision
        return nn.Identity()  # Placeholder

    def forward(self, images: torch.Tensor,
                text_queries: List[str]) -> Dict[str, torch.Tensor]:
        """
        Ground text queries in visual space.

        Args:
            images: Batch of images [B, C, H, W]
            text_queries: List of text queries [B]

        Returns:
            Dictionary with grounding results
        """
        batch_size = images.size(0)

        # Get CLIP embeddings
        visual_features = self.clip_model.get_image_features(pixel_values=images)
        text_features = self.encode_texts(text_queries)

        # Compute similarity between visual and text features
        logits_per_image = self.clip_model.logits_per_image  # [B, B]
        logits_per_text = self.clip_model.logits_per_text   # [B, B]

        # Perform grounding
        grounding_results = self.perform_grounding(
            visual_features, text_features, images
        )

        return grounding_results

    def encode_texts(self, text_queries: List[str]) -> torch.Tensor:
        """Encode text queries using CLIP text encoder."""
        inputs = self.processor(text=text_queries, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)

        return text_features

    def perform_grounding(self, visual_features: torch.Tensor,
                         text_features: torch.Tensor,
                         images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform vision-language grounding.
        """
        batch_size = visual_features.size(0)

        # Compute similarity matrix
        similarity = torch.matmul(visual_features, text_features.t())  # [B, B]

        # Apply grounding head
        grounded_features = self.grounding_head(visual_features)  # [B, grounding_dim]

        # Compute spatial localization (simplified)
        # In practice, this would use attention maps or detection bounding boxes
        localization = self.localization_head(visual_features)  # [B, 4]

        # Normalize localization to image coordinates
        localization_normalized = torch.sigmoid(localization)  # [0, 1] range

        return {
            'similarity_matrix': similarity,
            'grounded_features': grounded_features,
            'spatial_locations': localization_normalized,
            'attention_maps': self.compute_attention_maps(visual_features, text_features)
        }

    def compute_attention_maps(self, visual_features: torch.Tensor,
                              text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention maps showing which visual regions attend to text.
        """
        # Compute attention weights
        attention_weights = torch.matmul(visual_features, text_features.t())
        attention_weights = F.softmax(attention_weights, dim=-1)

        return attention_weights


class GroundingPostProcessor:
    """
    Post-process grounding results for robotics applications.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def process_grounding_results(self, results: Dict[str, torch.Tensor],
                                 image_size: Tuple[int, int]) -> List[Dict]:
        """
        Process grounding results into actionable information for robotics.

        Args:
            results: Output from VisionLanguageGrounding
            image_size: (height, width) of input image

        Returns:
            List of grounded objects with bounding boxes and confidence
        """
        spatial_locations = results['spatial_locations']  # [B, 4] normalized coords
        similarity_matrix = results['similarity_matrix']  # [B, B]

        grounded_objects = []

        for i in range(spatial_locations.size(0)):
            # Get similarity scores for this image-text pair
            text_similarities = similarity_matrix[i, :]  # [B]

            # Find most similar text
            max_sim_idx = torch.argmax(text_similarities)
            max_similarity = text_similarities[max_sim_idx]

            if max_similarity.item() > self.confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                norm_coords = spatial_locations[i]
                x_center = norm_coords[0].item()
                y_center = norm_coords[1].item()
                width = norm_coords[2].item()
                height = norm_coords[3].item()

                # Convert to bounding box format [x_min, y_min, x_max, y_max]
                x_min = int((x_center - width/2) * image_size[1])
                y_min = int((y_center - height/2) * image_size[0])
                x_max = int((x_center + width/2) * image_size[1])
                y_max = int((y_center + height/2) * image_size[0])

                grounded_object = {
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': max_similarity.item(),
                    'text_query': f'object_{max_sim_idx}',  # In practice, use actual text
                    'center': [(x_min + x_max) / 2, (y_min + y_max) / 2]
                }

                grounded_objects.append(grounded_object)

        return grounded_objects

    def filter_by_size(self, objects: List[Dict],
                      min_area: float = 0.01, max_area: float = 0.9) -> List[Dict]:
        """
        Filter grounded objects by size constraints.
        """
        filtered_objects = []

        for obj in objects:
            bbox = obj['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # Normalize by image area
            image_area = 1.0  # Already normalized
            normalized_area = area  # This would be normalized in practice

            if min_area <= normalized_area <= max_area:
                filtered_objects.append(obj)

        return filtered_objects


# Example usage
if __name__ == "__main__":
    # Initialize grounding module
    grounding_module = VisionLanguageGrounding()
    postprocessor = GroundingPostProcessor(confidence_threshold=0.5)

    # Example inputs
    images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    texts = ["red cup", "blue bottle"]  # Corresponding text queries

    # Perform grounding
    results = grounding_module(images, texts)

    # Process results
    grounded_objects = postprocessor.process_grounding_results(
        results, image_size=(224, 224)
    )

    print(f"Found {len(grounded_objects)} grounded objects")
    for i, obj in enumerate(grounded_objects):
        print(f"Object {i}: {obj['bbox']}, conf: {obj['confidence']:.3f}")
```

## Exercises

1. Implement the basic Vision-Language-Action fusion architecture
2. Create a ROS 2 node that integrates multi-modal fusion
3. Develop a vision-language grounding system for object detection
4. Implement sensor fusion with different modalities (LiDAR, IMU, cameras)
5. Test the system with real sensor data from a robot
6. Evaluate fusion performance under different sensor failure conditions
7. Optimize the fusion pipeline for real-time performance
8. Create a multi-modal dataset for training the fusion model

## References

1. CLIP: Learning Transferable Visual Models From Natural Language Supervision: https://arxiv.org/abs/2103.00020
2. DETR: End-to-End Object Detection with Transformers: https://arxiv.org/abs/2005.12872
3. ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision: https://arxiv.org/abs/2102.03334
4. ROS 2 Multi-robot Systems: http://wiki.ros.org/multirobot

## Further Reading

- Attention mechanisms in multi-modal learning
- Cross-modal transformers for robotics
- Uncertainty quantification in sensor fusion
- Real-time optimization of multi-modal pipelines
- Integration with reinforcement learning for robotic control
- Privacy considerations in multi-modal data processing