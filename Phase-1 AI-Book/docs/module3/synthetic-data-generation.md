---
sidebar_position: 32
---

# Synthetic Data Generation Pipelines for Perception Training

## Learning Objectives
By the end of this module, students will be able to:
- Design synthetic data generation pipelines for robotics perception
- Configure photorealistic rendering in Isaac Sim for synthetic datasets
- Implement domain randomization techniques to improve model generalization
- Generate diverse training datasets with accurate annotations
- Validate synthetic data quality against real-world benchmarks
- Optimize generation pipelines for computational efficiency

## Theory

### Synthetic Data in Robotics

Synthetic data generation is crucial for robotics perception training, especially when real-world data is scarce, expensive to collect, or dangerous to obtain. In robotics, synthetic data enables:

- Training perception models without collecting real data
- Generating edge cases that rarely occur in reality
- Creating perfectly labeled datasets with ground truth annotations
- Controlling environmental conditions systematically
- Scaling data collection without physical constraints

### Key Concepts

#### 1. Domain Randomization
Domain randomization is a technique where simulation parameters are randomly varied during training to make models robust to domain shift. This includes:
- Randomizing object appearances (textures, colors, materials)
- Varying lighting conditions (intensity, direction, color temperature)
- Changing camera parameters (FOV, noise, distortion)
- Modifying environmental properties (weather, time of day)

#### 2. Photorealistic Rendering
Modern synthetic data generation relies on photorealistic rendering to bridge the sim-to-real gap:
- Physically-based rendering (PBR) for realistic materials
- Global illumination for accurate lighting
- Realistic sensor simulation (noise, blur, distortion)
- High-resolution output for fine-grained details

#### 3. Ground Truth Generation
Synthetic environments provide perfect ground truth:
- Pixel-perfect segmentation masks
- Accurate 3D bounding boxes
- Precise pose information
- Depth maps and point clouds
- Optical flow and scene flow

### Isaac Sim Synthetic Data Tools

Isaac Sim provides several tools for synthetic data generation:
- **Synthetic Data Extension**: Generates 2D and 3D annotations
- **Replicator**: Procedural content generation framework
- **Differential Gaussian Splatting**: Novel view synthesis
- **PhysX Integration**: Accurate physics simulation for dynamic scenes

## Implementation

### Prerequisites
- Isaac Sim installed with Omniverse support
- RTX GPU with adequate VRAM (minimum 8GB, recommended 24GB+)
- Python 3.8 or higher
- Understanding of USD (Universal Scene Description)

### 1. Basic Synthetic Data Pipeline

#### Setting up the Environment
```python
# setup_synthetic_pipeline.py
import omni
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np
import cv2
import os

# Initialize Isaac Sim
config = {
    "width": 640,
    "height": 480,
    "output_dir": "./synthetic_dataset",
    "num_samples": 1000,
    "domain_randomization": True
}

def setup_replication_environment():
    """Initialize the synthetic data generation environment."""
    # Enable synthetic data extension
    omni.synthetic.aperture.activate()

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "images"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "labels"), exist_ok=True)

    # Set up camera
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.sensor import Camera

    # Create camera prim
    camera_prim = XFormPrim(
        prim_path="/World/Camera",
        translation=np.array([1.0, 0.0, 1.5]),
        orientation=np.array([0.707, 0, -0.707, 0])  # Look at origin
    )

    # Attach camera sensor
    camera = Camera(
        prim_path="/World/Camera",
        frequency=20,
        resolution=(config["width"], config["height"])
    )

    return camera

def setup_replicator_graph(camera):
    """Set up the replicator graph for data generation."""
    # Create a render product for the camera
    render_product = rep.create.render_product(camera, (config["width"], config["height"]))

    # Define the output annotations
    with rep.new.graph() as graph:
        # RGB output
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(render_product)

        # Semantic segmentation output
        seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        seg.attach(render_product)

        # Bounding box 2D output
        bbox_2d = rep.AnnotatorRegistry.get_annotator("bbox_2d_tight")
        bbox_2d.attach(render_product)

        # Depth output
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        depth.attach(render_product)

        # Write data to disk
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=config["output_dir"],
            rgb=True,
            semantic_segmentation=True,
            bbox_2d_tight=True,
            distance_to_camera=True
        )
        writer.attach([render_product])

    return graph
```

#### 2. Domain Randomization Setup
```python
def setup_domain_randomization():
    """Configure domain randomization parameters."""

    # Randomize lighting
    with rep.terrain.lighting():
        # Randomize dome light
        dome_light = rep.create.light(
            light_type="Dome",
            rotation=rep.distribution.uniform((-90, -90, -90), (90, 90, 90)),
            parameters={"color": rep.distribution.uniform((0.2, 0.2, 0.2), (1.0, 1.0, 1.0))}
        )

        # Randomize directional lights
        directional_light = rep.create.light(
            light_type="Distant",
            position=rep.distribution.uniform((-10, -10, 10), (10, 10, 10)),
            parameters={"intensity": rep.distribution.uniform(500, 1500)}
        )

    # Randomize materials
    def randomize_materials():
        # Find all materials in the scene
        materials = rep.utils.get_usd_omnipose_materials()

        with rep.randomizer.material_variations(materials):
            # Randomize diffuse color
            diffuse_color = rep.distribution.uniform((0.1, 0.1, 0.1, 1.0), (1.0, 1.0, 1.0, 1.0))

            # Randomize roughness
            roughness = rep.distribution.uniform(0.1, 1.0)

            # Randomize metallic
            metallic = rep.distribution.uniform(0.0, 1.0)

            return diffuse_color, roughness, metallic

    # Randomize textures
    def randomize_textures():
        # Apply random textures to objects
        with rep.randomizer.material_variations():
            # Randomize albedo maps
            albedo_path = rep.distribution.choice([
                "omniverse://localhost/NVIDIA/Assets/Materials/Base/ArchVis/Textures/concrete_01_diffuse.png",
                "omniverse://localhost/NVIDIA/Assets/Materials/Base/ArchVis/Textures/wood_01_diffuse.png",
                "omniverse://localhost/NVIDIA/Assets/Materials/Base/ArchVis/Textures/metal_01_diffuse.png"
            ])

            # Randomize normal maps
            normal_path = rep.distribution.choice([
                "omniverse://localhost/NVIDIA/Assets/Materials/Base/ArchVis/Textures/concrete_01_normal.png",
                "omniverse://localhost/NVIDIA/Assets/Materials/Base/ArchVis/Textures/wood_01_normal.png",
                "omniverse://localhost/NVIDIA/Assets/Materials/Base/ArchVis/Textures/metal_01_normal.png"
            ])

            return albedo_path, normal_path

    # Randomize camera parameters
    def randomize_camera():
        # Randomize camera position around the scene
        camera_positions = rep.distribution.uniform((-2.0, -2.0, 1.0), (2.0, 2.0, 3.0))

        # Randomize field of view
        fov = rep.distribution.uniform(30, 60)  # degrees

        return camera_positions, fov

    return randomize_materials, randomize_textures, randomize_camera
```

#### 3. Advanced Object Placement
```python
def setup_object_placement():
    """Set up procedural object placement with physics."""

    # Define object assets to use
    object_assets = [
        "omniverse://localhost/NVIDIA/Assets/Isaac/Props/KIT/Axis_Cube.usd",
        "omniverse://localhost/NVIDIA/Assets/Isaac/Props/Blocks/block_2x2x2_magenta.usd",
        "omniverse://localhost/NVIDIA/Assets/Isaac/Props/Blocks/block_4x1x1_blue.usd",
        "omniverse://localhost/NVIDIA/Assets/Isaac/Props/Blocks/block_8x1x2_red.usd"
    ]

    def random_object_placer():
        # Randomly select an object asset
        selected_asset = rep.distribution.choice(object_assets)

        # Randomize position within bounds
        position = rep.distribution.uniform((-1.0, -1.0, 0.1), (1.0, 1.0, 2.0))

        # Randomize rotation
        rotation = rep.distribution.uniform((0, 0, 0), (360, 360, 360))

        # Randomize scale
        scale = rep.distribution.uniform((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))

        # Create the object
        with rep.randomization.instantiate(selected_asset, position=position, rotation=rotation, scale=scale):
            return selected_asset, position, rotation, scale

    # Create a trigger to place objects randomly
    trigger = rep.trigger.on_frame(num_frames=1)

    with trigger:
        random_object_placer()

    return trigger

def setup_environment_scenes():
    """Create diverse environment scenes."""

    # Define different scene types
    scene_configs = {
        "indoor_office": {
            "lighting": "indoor",
            "objects": ["desk", "chair", "computer", "books"],
            "materials": ["wood", "metal", "plastic"],
            "background": "office"
        },
        "indoor_apartment": {
            "lighting": "warm",
            "objects": ["sofa", "table", "lamp", "plants"],
            "materials": ["fabric", "wood", "ceramic"],
            "background": "apartment"
        },
        "outdoor_urban": {
            "lighting": "outdoor",
            "objects": ["bench", "trash_bin", "lamppost"],
            "materials": ["concrete", "metal", "grass"],
            "background": "urban"
        }
    }

    def scene_selector():
        # Randomly select a scene configuration
        selected_scene = rep.distribution.choice(list(scene_configs.keys()))
        return selected_scene

    return scene_configs
```

#### 4. Synthetic Dataset Generator
```python
import json
from PIL import Image
import numpy as np

class SyntheticDatasetGenerator:
    """Main class for synthetic dataset generation."""

    def __init__(self, config):
        self.config = config
        self.dataset_stats = {
            "total_samples": 0,
            "scene_types": {},
            "object_counts": {},
            "annotation_quality": 0.0
        }

    def generate_sample(self, sample_id):
        """Generate a single synthetic data sample."""

        # Update scene with domain randomization
        self.apply_domain_randomization()

        # Capture all annotations
        annotations = self.capture_annotations(sample_id)

        # Save sample data
        self.save_sample(sample_id, annotations)

        # Update statistics
        self.update_statistics(annotations)

        return annotations

    def apply_domain_randomization(self):
        """Apply domain randomization to the current scene."""

        # Randomize lighting conditions
        self.randomize_lighting()

        # Randomize materials and textures
        self.randomize_materials()

        # Randomize camera parameters
        self.randomize_camera()

        # Randomize object placements
        self.randomize_objects()

    def randomize_lighting(self):
        """Randomize lighting conditions in the scene."""
        # Get current dome light
        dome_light_path = "/World/DomeLight"

        # Randomize intensity and color
        intensity = np.random.uniform(500, 2000)
        color_temp = np.random.uniform(3000, 8000)  # Kelvin

        # Convert color temperature to RGB
        rgb_color = self.color_temperature_to_rgb(color_temp)

        # Apply changes
        from omni import ui
        stage = omni.usd.get_context().get_stage()

        # Update dome light properties
        # This would typically involve modifying USD prim attributes
        pass

    def color_temperature_to_rgb(self, temp_kelvin):
        """Convert color temperature in Kelvin to RGB values."""
        temp = temp_kelvin / 100.0

        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            red = max(0, min(255, red))

        if temp <= 66:
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)

        green = max(0, min(255, green))

        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
            blue = max(0, min(255, blue))

        return np.array([red/255.0, green/255.0, blue/255.0])

    def capture_annotations(self, sample_id):
        """Capture all annotations for the current frame."""

        annotations = {
            "sample_id": sample_id,
            "timestamp": sample_id / self.config["sampling_rate"],
            "camera_params": self.get_camera_params(),
            "rgb": f"images/rgb_{sample_id:06d}.png",
            "depth": f"images/depth_{sample_id:06d}.exr",
            "segmentation": f"labels/seg_{sample_id:06d}.png",
            "bbox_2d": [],
            "bbox_3d": [],
            "poses": {},
            "metadata": {
                "lighting_condition": self.current_lighting,
                "scene_type": self.current_scene,
                "domain_randomization_params": self.dr_params
            }
        }

        # Get 2D bounding boxes
        bbox_2d_data = self.get_bbox_2d_data()
        annotations["bbox_2d"] = bbox_2d_data

        # Get 3D bounding boxes
        bbox_3d_data = self.get_bbox_3d_data()
        annotations["bbox_3d"] = bbox_3d_data

        # Get object poses
        pose_data = self.get_pose_data()
        annotations["poses"] = pose_data

        return annotations

    def save_sample(self, sample_id, annotations):
        """Save a sample with all its annotations."""

        # Save RGB image
        rgb_img = self.get_current_rgb_image()
        rgb_path = os.path.join(self.config["output_dir"], annotations["rgb"])
        Image.fromarray(rgb_img).save(rgb_path)

        # Save depth image
        depth_img = self.get_current_depth_image()
        depth_path = os.path.join(self.config["output_dir"], annotations["depth"])
        Image.fromarray((depth_img * 65535).astype(np.uint16)).save(depth_path)

        # Save segmentation
        seg_img = self.get_current_segmentation()
        seg_path = os.path.join(self.config["output_dir"], annotations["segmentation"])
        Image.fromarray(seg_img).save(seg_path)

        # Save annotations as JSON
        anno_path = os.path.join(
            self.config["output_dir"],
            "labels",
            f"annotations_{sample_id:06d}.json"
        )
        with open(anno_path, 'w') as f:
            json.dump(annotations, f, indent=2)

    def generate_dataset(self):
        """Generate the complete synthetic dataset."""

        print(f"Generating {self.config['num_samples']} samples...")

        for i in range(self.config["num_samples"]):
            print(f"Generating sample {i+1}/{self.config['num_samples']}")

            # Generate one sample
            annotations = self.generate_sample(i)

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"Progress: {(i+1)/self.config['num_samples']*100:.1f}%")

        # Save dataset statistics
        self.save_statistics()

        print(f"Dataset generation completed! Saved to {self.config['output_dir']}")
        print(f"Statistics: {self.dataset_stats}")

    def save_statistics(self):
        """Save dataset statistics."""
        stats_path = os.path.join(self.config["output_dir"], "dataset_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.dataset_stats, f, indent=2)

# Usage example
if __name__ == "__main__":
    config = {
        "num_samples": 5000,
        "output_dir": "./synthetic_robotics_dataset",
        "width": 640,
        "height": 480,
        "sampling_rate": 10,  # Hz
        "domain_randomization": True,
        "scene_complexity": "medium"  # low, medium, high
    }

    generator = SyntheticDatasetGenerator(config)
    generator.generate_dataset()
```

### 5. Perception Training Data Preparation
```python
import yaml
import shutil
from pathlib import Path

def prepare_training_data(config):
    """Prepare synthetic data for perception training."""

    # Create standard dataset structure
    dataset_path = Path(config["output_dir"])
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    test_path = dataset_path / "test"

    for path in [train_path, val_path, test_path]:
        (path / "images").mkdir(parents=True, exist_ok=True)
        (path / "labels").mkdir(parents=True, exist_ok=True)

    # Split data into train/val/test (80/10/10)
    all_samples = list((dataset_path / "images").glob("*.png"))
    np.random.shuffle(all_samples)

    num_train = int(len(all_samples) * 0.8)
    num_val = int(len(all_samples) * 0.1)

    train_samples = all_samples[:num_train]
    val_samples = all_samples[num_train:num_train+num_val]
    test_samples = all_samples[num_train+num_val:]

    # Copy files to appropriate directories
    def copy_samples(samples, split_path):
        for img_path in samples:
            # Copy image
            shutil.copy2(img_path, split_path / "images")

            # Copy corresponding annotation
            anno_path = dataset_path / "labels" / f"annotations_{img_path.stem[4:]}.json"
            if anno_path.exists():
                shutil.copy2(anno_path, split_path / "labels")

    copy_samples(train_samples, train_path)
    copy_samples(val_samples, val_path)
    copy_samples(test_samples, test_path)

    # Create dataset configuration file
    dataset_config = {
        'path': str(dataset_path.absolute()),
        'train': str((train_path / 'images').relative_to(dataset_path)),
        'val': str((val_path / 'images').relative_to(dataset_path)),
        'test': str((test_path / 'images').relative_to(dataset_path)),
        'nc': config.get('num_classes', 10),  # Number of classes
        'names': config.get('class_names', [f'class_{i}' for i in range(config.get('num_classes', 10))])
    }

    with open(dataset_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)

    print(f"Training data prepared at {dataset_path}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

# Example usage
training_config = {
    "output_dir": "./synthetic_robotics_dataset",
    "num_classes": 15,
    "class_names": [
        "person", "chair", "table", "sofa", "lamp",
        "plant", "cup", "bottle", "box", "monitor",
        "keyboard", "mouse", "book", "phone", "robot"
    ]
}

prepare_training_data(training_config)
```

### 6. Quality Assurance and Validation
```python
import matplotlib.pyplot as plt
from scipy import ndimage
import seaborn as sns

def validate_synthetic_data_quality(dataset_path):
    """Validate the quality of generated synthetic data."""

    dataset_path = Path(dataset_path)

    # Load sample annotations
    annotation_files = list((dataset_path / "labels").glob("*.json"))

    if not annotation_files:
        print("No annotation files found!")
        return

    # Sample some annotations for analysis
    sample_annotations = np.random.choice(annotation_files, min(100, len(annotation_files)), replace=False)

    # Analyze annotation quality
    bbox_sizes = []
    object_counts = []
    depth_ranges = []

    for anno_file in sample_annotations:
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        # Analyze bounding boxes
        bboxes = anno.get("bbox_2d", [])
        for bbox in bboxes:
            width = bbox["xmax"] - bbox["xmin"]
            height = bbox["ymax"] - bbox["ymin"]
            bbox_sizes.append([width, height])

        object_counts.append(len(bboxes))

        # Analyze depth
        depth_path = dataset_path / anno["depth"]
        if depth_path.exists():
            depth_img = np.array(Image.open(depth_path))
            depth_ranges.append([np.min(depth_img), np.max(depth_img)])

    # Generate quality reports
    generate_quality_report(bbox_sizes, object_counts, depth_ranges, dataset_path)

def generate_quality_report(bbox_sizes, object_counts, depth_ranges, output_path):
    """Generate a quality report for synthetic data."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Bounding box size distribution
    if bbox_sizes:
        bbox_sizes = np.array(bbox_sizes)
        axes[0, 0].scatter(bbox_sizes[:, 0], bbox_sizes[:, 1], alpha=0.6)
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].set_title('Bounding Box Size Distribution')

    # Plot 2: Object count distribution
    if object_counts:
        axes[0, 1].hist(object_counts, bins=20, edgecolor='black')
        axes[0, 1].set_xlabel('Number of Objects per Image')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Object Count Distribution')

    # Plot 3: Depth range distribution
    if depth_ranges:
        depth_ranges = np.array(depth_ranges)
        axes[1, 0].plot(depth_ranges[:, 0], label='Min Depth', alpha=0.7)
        axes[1, 0].plot(depth_ranges[:, 1], label='Max Depth', alpha=0.7)
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Depth (m)')
        axes[1, 0].set_title('Depth Range Across Samples')
        axes[1, 0].legend()

    # Plot 4: Quality metrics
    metrics = {
        'Avg Objects per Image': np.mean(object_counts) if object_counts else 0,
        'Avg BBox Width': np.mean([s[0] for s in bbox_sizes]) if bbox_sizes else 0,
        'Avg BBox Height': np.mean([s[1] for s in bbox_sizes]) if bbox_sizes else 0,
        'Avg Scene Depth': np.mean([np.mean(dr) for dr in depth_ranges]) if depth_ranges else 0
    }

    axes[1, 1].bar(metrics.keys(), metrics.values())
    axes[1, 1].set_title('Quality Metrics')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path / "quality_report.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed statistics
    stats = {
        "total_samples": len(bbox_sizes),
        "average_objects_per_image": np.mean(object_counts) if object_counts else 0,
        "std_objects_per_image": np.std(object_counts) if object_counts else 0,
        "average_bbox_width": np.mean([s[0] for s in bbox_sizes]) if bbox_sizes else 0,
        "average_bbox_height": np.mean([s[1] for s in bbox_sizes]) if bbox_sizes else 0,
        "min_depth_range": np.min([r[0] for r in depth_ranges]) if depth_ranges else 0,
        "max_depth_range": np.max([r[1] for r in depth_ranges]) if depth_ranges else float('inf'),
        "average_depth_span": np.mean([r[1]-r[0] for r in depth_ranges]) if depth_ranges else 0
    }

    with open(output_path / "quality_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("Quality report generated successfully!")
    print(f"Statistics: {stats}")

# Validate the generated dataset
validate_synthetic_data_quality("./synthetic_robotics_dataset")
```

## Exercises

1. Create a synthetic data pipeline for object detection with 10 different object classes
2. Implement domain randomization with at least 5 different parameters (lighting, materials, textures, camera, objects)
3. Generate a dataset of 1000 images with perfect annotations for semantic segmentation
4. Validate the quality of your synthetic dataset using the provided tools
5. Compare synthetic vs real data performance on a simple object detection model
6. Optimize your generation pipeline for speed while maintaining quality
7. Create different scene types (indoor, outdoor, office, home) in your dataset

## References

1. NVIDIA Isaac Sim Synthetic Data Documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/features/synthetic_data/index.html
2. Domain Randomization Paper: https://arxiv.org/abs/1703.06907
3. NVIDIA Replicator Documentation: https://docs.omniverse.nvidia.com/py/replicator/
4. Synthetic Data for Computer Vision: https://research.nvidia.com/publication/2021-06_Synthetic-Data-Generation

## Further Reading

- Advanced USD composition for complex scenes
- Physics-aware synthetic data generation
- Generative Adversarial Networks for synthetic data enhancement
- Active learning with synthetic and real data combination
- Transfer learning from synthetic to real domains
- Evaluation metrics for synthetic data quality
- Large-scale distributed synthetic data generation