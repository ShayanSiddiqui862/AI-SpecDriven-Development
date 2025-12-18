---
sidebar_position: 35
---

# Verification: Object Detection and Navigation Stacks

## Learning Objectives
By the end of this module, students will be able to:
- Validate object detection accuracy in simulated and real environments
- Verify navigation stack performance with bipedal robot constraints
- Test perception-navigation integration in complex scenarios
- Evaluate system robustness under various conditions
- Debug and troubleshoot perception and navigation issues

## Theory

### Object Detection Validation

#### 1. Performance Metrics
- **Precision**: TP/(TP+FP) - Fraction of positive predictions that are correct
- **Recall**: TP/(TP+FN) - Fraction of actual positives that are correctly identified
- **mAP (mean Average Precision)**: Average precision across all classes
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth bounding boxes
- **FPS (Frames Per Second)**: Processing speed requirement for real-time applications

#### 2. Validation Approaches
- **Synthetic-to-Real Transfer**: Validating performance from synthetic training to real data
- **Cross-Environment Validation**: Testing performance across different lighting/scene conditions
- **Edge Case Testing**: Evaluating performance on rare or challenging scenarios
- **Robustness Testing**: Checking performance under sensor noise, occlusions, etc.

### Navigation Stack Validation

#### 1. Path Planning Metrics
- **Success Rate**: Percentage of successful navigation attempts
- **Path Optimality**: Ratio of planned path length to optimal path length
- **Computational Efficiency**: Planning time and resource usage
- **Dynamic Obstacle Avoidance**: Ability to handle moving obstacles

#### 2. Navigation Performance Indicators
- **Trajectory Tracking Accuracy**: Deviation from planned path
- **Collision Avoidance**: Ability to avoid obstacles safely
- **Recovery Behavior**: Performance when stuck or lost
- **Balance Maintenance**: Stability during navigation for bipedal robots

### Integration Challenges

#### 1. Perception-Navigation Pipeline
- **Timing Synchronization**: Ensuring perception outputs align with navigation updates
- **Coordinate Frame Alignment**: Proper transformation between perception and navigation frames
- **Uncertainty Propagation**: Handling perception uncertainty in navigation decisions
- **Feedback Loops**: Using navigation success/failure to improve perception

#### 2. Real-time Constraints
- **Processing Latency**: Perception processing time affecting navigation decisions
- **Update Frequency**: Required update rates for stable navigation
- **Resource Competition**: Shared computational resources between perception and navigation

## Implementation

### Prerequisites
- ROS 2 Humble with perception and navigation packages
- Object detection models trained on relevant datasets
- Validated navigation stack configuration
- Testing environments (simulated and potentially real)

### 1. Object Detection Validation Framework

#### Detection Validation Node
```python
#!/usr/bin/env python3
"""
Object detection validation framework for robotics applications.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
import json
import os
from collections import defaultdict
import time

class ObjectDetectionValidator(Node):
    """
    Validates object detection performance in robotics scenarios.
    """

    def __init__(self):
        super().__init__('object_detection_validator')

        # Parameters
        self.declare_parameter('validation_dataset_path', '/path/to/validation/dataset')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('results_output_path', '/tmp/detection_results')

        self.validation_dataset_path = self.get_parameter('validation_dataset_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.results_output_path = self.get_parameter('results_output_path').value

        # CV Bridge
        self.bridge = CvBridge()

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detector/detections',
            self.detection_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.validation_pub = self.create_publisher(
            Detection2DArray,
            '/validation/detections',
            10
        )

        # State variables
        self.current_image = None
        self.current_detections = None
        self.ground_truth_available = True
        self.validation_results = {
            'precision': [],
            'recall': [],
            'mAP': [],
            'inference_time': [],
            'fps': []
        }

        # Timers
        self.validation_timer = self.create_timer(1.0, self.run_validation_cycle)

        self.get_logger().info('Object detection validator initialized')

    def image_callback(self, msg):
        """Process incoming image."""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def detection_callback(self, msg):
        """Process incoming detections."""
        self.current_detections = msg

    def run_validation_cycle(self):
        """Run periodic validation cycle."""
        if self.current_image is not None and self.current_detections is not None:
            # Perform validation
            results = self.validate_current_frame()

            # Update metrics
            if results['precision'] is not None:
                self.validation_results['precision'].append(results['precision'])
            if results['recall'] is not None:
                self.validation_results['recall'].append(results['recall'])
            if results['inference_time'] is not None:
                self.validation_results['inference_time'].append(results['inference_time'])

            # Log results
            self.get_logger().info(
                f'Detection Validation - Precision: {results["precision"]:.3f}, '
                f'Recall: {results["recall"]:.3f}, '
                f'Inference Time: {results["inference_time"]:.3f}s'
            )

            # Reset for next frame
            self.current_detections = None

    def validate_current_frame(self):
        """Validate current frame detections against ground truth."""
        if not self.ground_truth_available:
            # If no ground truth, use temporal consistency validation
            return self.validate_temporal_consistency()

        # Get ground truth for current frame (would come from dataset)
        ground_truth = self.get_ground_truth_for_frame()

        if ground_truth is None:
            return {'precision': None, 'recall': None, 'inference_time': None}

        # Calculate metrics
        precision, recall = self.calculate_precision_recall(
            self.current_detections, ground_truth)

        # Calculate mAP if possible
        mAP = self.calculate_mean_average_precision(
            self.current_detections, ground_truth)

        # Measure inference time (approximate)
        inference_time = self.estimate_inference_time()

        return {
            'precision': precision,
            'recall': recall,
            'mAP': mAP,
            'inference_time': inference_time
        }

    def calculate_precision_recall(self, detections, ground_truth):
        """Calculate precision and recall for detections."""
        # Convert detections to format for comparison
        det_boxes = []
        det_scores = []
        det_labels = []

        for detection in detections.detections:
            bbox = detection.bbox
            box = [bbox.center.x - bbox.size_x/2, bbox.center.y - bbox.size_y/2,
                   bbox.center.x + bbox.size_x/2, bbox.center.y + bbox.size_y/2]
            det_boxes.append(box)

            # Get highest scoring hypothesis
            if detection.results:
                max_score = max(h.score for h in detection.results)
                det_scores.append(max_score)
                max_idx = np.argmax([h.score for h in detection.results])
                det_labels.append(detection.results[max_idx].id)
            else:
                det_scores.append(0.0)
                det_labels.append('')

        gt_boxes = []
        gt_labels = []

        for gt_obj in ground_truth:
            gt_boxes.append(gt_obj['bbox'])
            gt_labels.append(gt_obj['label'])

        # Calculate IoU for each detection-ground truth pair
        tp = 0  # True positives
        fp = 0  # False positives

        matched_gt = set()

        # Sort detections by confidence
        sorted_indices = sorted(range(len(det_scores)),
                               key=lambda i: det_scores[i], reverse=True)

        for det_idx in sorted_indices:
            if det_scores[det_idx] < self.confidence_threshold:
                continue

            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = self.calculate_iou(det_boxes[det_idx], gt_box)

                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx != -1:
                # Check if labels match
                if det_labels[det_idx] == gt_labels[best_gt_idx]:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)  # False negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return precision, recall

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        # box format: [x_min, y_min, x_max, y_max]
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def calculate_mean_average_precision(self, detections, ground_truth):
        """Calculate mean Average Precision."""
        # Implementation of mAP calculation
        # This is a simplified version - in practice, you'd use a more sophisticated approach
        # that considers different confidence thresholds

        classes = set()
        for gt in ground_truth:
            classes.add(gt['label'])
        for det in detections.detections:
            if det.results:
                classes.add(det.results[0].id)

        aps = []
        for class_name in classes:
            ap = self.calculate_average_precision_for_class(
                detections, ground_truth, class_name)
            if ap is not None:
                aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def calculate_average_precision_for_class(self, detections, ground_truth, class_name):
        """Calculate AP for a specific class."""
        # Filter detections and ground truth for this class
        class_dets = []
        for det in detections.detections:
            if det.results and det.results[0].id == class_name:
                class_dets.append({
                    'bbox': det.bbox,
                    'score': det.results[0].score
                })

        class_gts = [gt for gt in ground_truth if gt['label'] == class_name]

        if len(class_gts) == 0:
            return None

        # Sort detections by score
        class_dets.sort(key=lambda x: x['score'], reverse=True)

        # Calculate precision-recall curve
        tp = np.zeros(len(class_dets))
        fp = np.zeros(len(class_dets))

        matched_gt = set()

        for det_idx, det in enumerate(class_dets):
            bbox = det['bbox']
            det_box = [bbox.center.x - bbox.size_x/2, bbox.center.y - bbox.size_y/2,
                      bbox.center.x + bbox.size_x/2, bbox.center.y + bbox.size_y/2]

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(class_gts):
                if gt_idx in matched_gt:
                    continue

                iou = self.calculate_iou(det_box, gt['bbox'])

                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx != -1:
                if best_gt_idx not in matched_gt:
                    tp[det_idx] = 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[det_idx] = 1
            else:
                fp[det_idx] = 1

        # Calculate cumulative tp and fp
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Calculate recall and precision
        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.any():
                p = np.max(precisions[mask])
            else:
                p = 0.0
            ap += p / 11.0

        return ap

    def estimate_inference_time(self):
        """Estimate inference time (placeholder implementation)."""
        # In a real implementation, this would measure actual processing time
        return 0.05  # 50ms as placeholder

    def get_ground_truth_for_frame(self):
        """Get ground truth annotations for current frame."""
        # In a real implementation, this would load ground truth from dataset
        # For now, return None to indicate ground truth not available
        return None

    def validate_temporal_consistency(self):
        """Validate temporal consistency of detections."""
        # Check if objects are detected consistently across frames
        # This is useful when ground truth is not available
        pass

    def save_validation_results(self):
        """Save validation results to file."""
        results_path = os.path.join(self.results_output_path, 'detection_validation_results.json')

        # Calculate aggregate metrics
        aggregate_results = {
            'overall_precision': np.mean(self.validation_results['precision']) if self.validation_results['precision'] else 0,
            'overall_recall': np.mean(self.validation_results['recall']) if self.validation_results['recall'] else 0,
            'overall_mAP': np.mean(self.validation_results['mAP']) if self.validation_results['mAP'] else 0,
            'avg_inference_time': np.mean(self.validation_results['inference_time']) if self.validation_results['inference_time'] else 0,
            'total_frames_processed': len(self.validation_results['precision']),
            'timestamp': time.time()
        }

        # Ensure output directory exists
        os.makedirs(self.results_output_path, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(aggregate_results, f, indent=2)

        self.get_logger().info(f'Validation results saved to {results_path}')
        return aggregate_results


def main(args=None):
    rclpy.init(args=args)

    validator = ObjectDetectionValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        # Save results on shutdown
        validator.save_validation_results()
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Navigation Stack Validator

#### Navigation Validation Node
```python
#!/usr/bin/env python3
"""
Navigation stack validation framework for robotics applications.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Float32
import numpy as np
import math
from scipy.spatial.distance import cdist
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import time
from collections import deque

class NavigationStackValidator(Node):
    """
    Validates navigation stack performance for robotics applications.
    """

    def __init__(self):
        super().__init__('navigation_stack_validator')

        # Parameters
        self.declare_parameter('validation_timeout', 300.0)  # 5 minutes
        self.declare_parameter('success_radius', 0.5)  # meters
        self.declare_parameter('min_path_length', 1.0)  # minimum path length to validate
        self.declare_parameter('results_output_path', '/tmp/navigation_validation_results')

        self.validation_timeout = self.get_parameter('validation_timeout').value
        self.success_radius = self.get_parameter('success_radius').value
        self.min_path_length = self.get_parameter('min_path_length').value
        self.results_output_path = self.get_parameter('results_output_path').value

        # State variables
        self.current_pose = None
        self.target_pose = None
        self.current_path = None
        self.navigation_start_time = None
        self.navigation_status = None
        self.path_execution_log = deque(maxlen=1000)
        self.collision_detected = False
        self.obstacle_distances = deque(maxlen=100)

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publishers
        self.validation_metrics_pub = self.create_publisher(
            Float32,
            '/navigation/validation_metrics',
            10
        )

        self.visualization_pub = self.create_publisher(
            MarkerArray,
            '/navigation/validation_visualization',
            10
        )

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Timers
        self.validation_timer = self.create_timer(0.1, self.validation_callback)
        self.metrics_timer = self.create_timer(1.0, self.publish_metrics)

        self.get_logger().info('Navigation stack validator initialized')

    def odom_callback(self, msg):
        """Process odometry updates."""
        self.current_pose = msg.pose.pose

        if self.navigation_start_time is not None and self.target_pose is not None:
            # Log current position and status
            self.path_execution_log.append({
                'timestamp': self.get_clock().now().nanoseconds * 1e-9,
                'position': msg.pose.pose.position,
                'distance_to_goal': self.calculate_distance_to_goal(
                    msg.pose.pose.position, self.target_pose.position),
                'path_progress': self.calculate_path_progress()
            })

    def path_callback(self, msg):
        """Process planned path."""
        self.current_path = msg.poses

    def scan_callback(self, msg):
        """Process laser scan for collision detection."""
        # Check for obstacles close to robot
        min_distance = min(msg.ranges) if msg.ranges else float('inf')
        self.obstacle_distances.append(min_distance)

        # Detect potential collision
        if min_distance < 0.3:  # 30cm threshold
            self.collision_detected = True

    def cmd_vel_callback(self, msg):
        """Process velocity commands for analysis."""
        # Log velocity commands to analyze navigation behavior
        current_time = self.get_clock().now().nanoseconds * 1e-9
        self.velocity_log.append({
            'timestamp': current_time,
            'linear_x': msg.linear.x,
            'angular_z': msg.angular.z
        })

    def validation_callback(self):
        """Main validation callback."""
        if self.navigation_start_time is not None and self.target_pose is not None:
            current_time = self.get_clock().now().nanoseconds * 1e-9
            elapsed_time = current_time - self.navigation_start_time

            # Check if navigation timed out
            if elapsed_time > self.validation_timeout:
                self.get_logger().warn('Navigation validation timed out')
                self.record_validation_result(success=False, reason='timeout')
                self.reset_validation_state()

            # Check if goal reached
            if (self.current_pose is not None and
                self.calculate_distance_to_goal(self.current_pose.position, self.target_pose.position) < self.success_radius):
                self.get_logger().info('Navigation goal reached successfully')
                self.record_validation_result(success=True, reason='goal_reached')
                self.reset_validation_state()

    def calculate_distance_to_goal(self, current_pos, target_pos):
        """Calculate Euclidean distance to goal."""
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        dz = target_pos.z - current_pos.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def calculate_path_progress(self):
        """Calculate progress along the planned path."""
        if not self.current_path or not self.current_pose:
            return 0.0

        # Find closest point on path to current position
        current_pos = self.current_pose.position
        min_dist = float('inf')
        progress = 0.0

        for i, pose in enumerate(self.current_path):
            path_pos = pose.pose.position
            dist = math.sqrt(
                (current_pos.x - path_pos.x)**2 +
                (current_pos.y - path_pos.y)**2
            )
            if dist < min_dist:
                min_dist = dist
                progress = i / len(self.current_path)

        return progress

    def calculate_path_efficiency(self):
        """Calculate path efficiency (actual vs optimal)."""
        if not self.path_execution_log or not self.current_path:
            return 1.0

        # Calculate actual distance traveled
        actual_distance = 0.0
        log_list = list(self.path_execution_log)
        for i in range(1, len(log_list)):
            prev_pos = log_list[i-1]['position']
            curr_pos = log_list[i]['position']
            actual_distance += math.sqrt(
                (curr_pos.x - prev_pos.x)**2 +
                (curr_pos.y - prev_pos.y)**2
            )

        # Calculate optimal path distance
        if len(self.current_path) >= 2:
            start_pos = self.current_path[0].pose.position
            end_pos = self.current_path[-1].pose.position
            optimal_distance = math.sqrt(
                (end_pos.x - start_pos.x)**2 +
                (end_pos.y - start_pos.y)**2
            )
        else:
            optimal_distance = actual_distance

        if optimal_distance > 0:
            return actual_distance / optimal_distance
        else:
            return 1.0

    def calculate_stability_metrics(self):
        """Calculate navigation stability metrics."""
        if len(self.velocity_log) < 10:
            return 1.0  # Not enough data

        # Calculate velocity smoothness (lower variance = more stable)
        linear_vels = [entry['linear_x'] for entry in self.velocity_log]
        angular_vels = [entry['angular_z'] for entry in self.velocity_log]

        linear_variance = np.var(linear_vels) if len(set(linear_vels)) > 1 else 0.0
        angular_variance = np.var(angular_vels) if len(set(angular_vels)) > 1 else 0.0

        # Stability score (higher is more stable, inverted variance)
        stability_score = 1.0 / (1.0 + linear_variance + angular_variance)
        return min(stability_score, 1.0)  # Cap at 1.0

    def calculate_obstacle_avoidance_metrics(self):
        """Calculate obstacle avoidance performance."""
        if not self.obstacle_distances:
            return 1.0

        # Calculate percentage of time with safe distances
        safe_distances = [d for d in self.obstacle_distances if d > 0.5]  # Safe distance: >50cm
        safety_percentage = len(safe_distances) / len(self.obstacle_distances)

        return safety_percentage

    def record_validation_result(self, success, reason):
        """Record validation result and calculate metrics."""
        if not self.path_execution_log:
            self.get_logger().warn('No path execution data recorded')
            return

        # Calculate metrics
        execution_time = self.path_execution_log[-1]['timestamp'] - self.navigation_start_time
        path_efficiency = self.calculate_path_efficiency()
        stability_score = self.calculate_stability_metrics()
        safety_score = self.calculate_obstacle_avoidance_metrics()

        # Calculate success rate based on multiple factors
        success_score = (
            0.4 * (1 if success else 0) +
            0.2 * (1.0 / max(path_efficiency, 0.1)) +  # Prefer efficient paths
            0.2 * stability_score +
            0.2 * safety_score
        )

        # Store results
        validation_result = {
            'success': success,
            'reason': reason,
            'execution_time': execution_time,
            'path_efficiency': path_efficiency,
            'stability_score': stability_score,
            'safety_score': safety_score,
            'success_score': success_score,
            'collision_detected': self.collision_detected,
            'timestamp': time.time()
        }

        self.validation_results.append(validation_result)

        # Log results
        self.get_logger().info(
            f'Navigation Validation Result:\n'
            f'  Success: {success} ({reason})\n'
            f'  Execution Time: {execution_time:.2f}s\n'
            f'  Path Efficiency: {path_efficiency:.2f}\n'
            f'  Stability Score: {stability_score:.2f}\n'
            f'  Safety Score: {safety_score:.2f}\n'
            f'  Overall Score: {success_score:.2f}'
        )

    def reset_validation_state(self):
        """Reset validation state for next test."""
        self.navigation_start_time = None
        self.target_pose = None
        self.current_path = None
        self.path_execution_log.clear()
        self.velocity_log.clear()
        self.obstacle_distances.clear()
        self.collision_detected = False

    def publish_metrics(self):
        """Publish current validation metrics."""
        if self.validation_results:
            latest_result = self.validation_results[-1]

            # Publish overall success score
            score_msg = Float32()
            score_msg.data = latest_result['success_score']
            self.validation_metrics_pub.publish(score_msg)

            # Publish visualization markers
            self.publish_validation_visualization()

    def publish_validation_visualization(self):
        """Publish visualization markers for validation results."""
        marker_array = MarkerArray()

        # Create marker for path
        path_marker = Marker()
        path_marker.header.frame_id = 'map'
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = 'navigation_validation'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD

        path_marker.pose.orientation.w = 1.0
        path_marker.scale.x = 0.05  # Line width

        if self.current_path:
            for pose in self.current_path:
                point = Point()
                point.x = pose.pose.position.x
                point.y = pose.pose.position.y
                point.z = pose.pose.position.z
                path_marker.points.append(point)

        # Color based on validation status
        if self.validation_results and self.validation_results[-1]['success']:
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
        else:
            path_marker.color.r = 1.0
            path_marker.color.g = 0.0
            path_marker.color.b = 0.0
        path_marker.color.a = 1.0

        marker_array.markers.append(path_marker)

        # Create marker for robot trajectory
        traj_marker = Marker()
        traj_marker.header.frame_id = 'map'
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.ns = 'navigation_validation'
        traj_marker.id = 1
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD

        traj_marker.pose.orientation.w = 1.0
        traj_marker.scale.x = 0.02  # Line width

        for log_entry in list(self.path_execution_log)[-100:]:  # Last 100 points
            point = Point()
            point.x = log_entry['position'].x
            point.y = log_entry['position'].y
            point.z = log_entry['position'].z
            traj_marker.points.append(point)

        traj_marker.color.r = 0.0
        traj_marker.color.g = 0.0
        traj_marker.color.b = 1.0
        traj_marker.color.a = 1.0

        marker_array.markers.append(traj_marker)

        self.visualization_pub.publish(marker_array)

    def run_comprehensive_validation(self, test_scenarios):
        """Run comprehensive validation with multiple test scenarios."""
        results_summary = {
            'total_tests': len(test_scenarios),
            'successful_tests': 0,
            'average_time': 0.0,
            'average_efficiency': 0.0,
            'average_stability': 0.0,
            'collision_rate': 0.0,
            'detailed_results': []
        }

        for i, scenario in enumerate(test_scenarios):
            self.get_logger().info(f'Running validation scenario {i+1}/{len(test_scenarios)}')

            # Set up scenario
            start_pose = scenario['start']
            goal_pose = scenario['goal']
            environment = scenario.get('environment', 'default')

            # Send navigation goal
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.pose.position.x = goal_pose[0]
            goal_msg.pose.pose.position.y = goal_pose[1]
            goal_msg.pose.pose.position.z = goal_pose[2] if len(goal_pose) > 2 else 0.0
            goal_msg.pose.pose.orientation.w = 1.0  # Simple orientation

            # Wait for server
            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Navigation server not available')
                continue

            # Send goal and wait for result
            future = self.nav_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)

            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Goal was rejected')
                continue

            # Monitor execution
            self.target_pose = goal_msg.pose.pose
            self.navigation_start_time = self.get_clock().now().nanoseconds * 1e-9

            # Wait for result with timeout
            result_future = goal_handle.get_result_async()
            timeout_time = time.time() + self.validation_timeout
            while time.time() < timeout_time:
                rclpy.spin_once(self, timeout_sec=0.1)
                if result_future.done():
                    break

            if result_future.done():
                result = result_future.result().result
                status = result.status
                success = (status == GoalStatus.STATUS_SUCCEEDED)

                # Record results
                if self.validation_results:
                    latest = self.validation_results[-1]
                    results_summary['detailed_results'].append(latest)
                    if latest['success']:
                        results_summary['successful_tests'] += 1
            else:
                # Timeout occurred
                self.record_validation_result(success=False, reason='timeout')
                if self.validation_results:
                    latest = self.validation_results[-1]
                    results_summary['detailed_results'].append(latest)

        # Calculate aggregate metrics
        if results_summary['detailed_results']:
            results_summary['average_time'] = np.mean([
                r['execution_time'] for r in results_summary['detailed_results']
            ])
            results_summary['average_efficiency'] = np.mean([
                r['path_efficiency'] for r in results_summary['detailed_results']
            ])
            results_summary['average_stability'] = np.mean([
                r['stability_score'] for r in results_summary['detailed_results']
            ])
            collision_count = sum(1 for r in results_summary['detailed_results'] if r['collision_detected'])
            results_summary['collision_rate'] = collision_count / len(results_summary['detailed_results'])

        # Save comprehensive results
        self.save_comprehensive_results(results_summary)

        self.get_logger().info(
            f'Comprehensive Validation Complete:\n'
            f'  Success Rate: {results_summary["successful_tests"]}/{results_summary["total_tests"]} '
            f'({results_summary["successful_tests"]/results_summary["total_tests"]*100:.1f}%)\n'
            f'  Avg Time: {results_summary["average_time"]:.2f}s\n'
            f'  Avg Efficiency: {results_summary["average_efficiency"]:.2f}\n'
            f'  Avg Stability: {results_summary["average_stability"]:.2f}\n'
            f'  Collision Rate: {results_summary["collision_rate"]:.2f}'
        )

        return results_summary

    def save_comprehensive_results(self, results_summary):
        """Save comprehensive validation results to file."""
        import json
        import os

        results_path = os.path.join(self.results_output_path, 'comprehensive_navigation_validation.json')
        os.makedirs(self.results_output_path, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        self.get_logger().info(f'Comprehensive results saved to {results_path}')


def main(args=None):
    rclpy.init(args=args)

    validator = NavigationStackValidator()

    # Example test scenarios (these would come from a test configuration)
    test_scenarios = [
        {
            'start': [0.0, 0.0, 0.0],
            'goal': [5.0, 0.0, 0.0],
            'environment': 'corridor'
        },
        {
            'start': [0.0, 0.0, 0.0],
            'goal': [3.0, 4.0, 0.0],
            'environment': 'open_space'
        },
        {
            'start': [0.0, 0.0, 0.0],
            'goal': [-2.0, -3.0, 0.0],
            'environment': 'cluttered'
        }
    ]

    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation(test_scenarios)

        # Continue with regular spinning to monitor ongoing navigation
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 3. Integration Validator

#### Perception-Navigation Integration Validator
```python
#!/usr/bin/env python3
"""
Integration validator for perception-navigation pipeline.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
import numpy as np
import math
import time
from collections import deque
import tf2_ros
from scipy.spatial.distance import cdist

class PerceptionNavigationValidator(Node):
    """
    Validates the integration between perception and navigation systems.
    """

    def __init__(self):
        super().__init__('perception_navigation_validator')

        # Parameters
        self.declare_parameter('validation_window', 10.0)  # seconds
        self.declare_parameter('detection_threshold', 0.5)  # confidence
        self.declare_parameter('navigation_timeout', 120.0)  # seconds
        self.declare_parameter('results_output_path', '/tmp/integration_validation')

        self.validation_window = self.get_parameter('validation_window').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.navigation_timeout = self.get_parameter('navigation_timeout').value
        self.results_output_path = self.get_parameter('results_output_path').value

        # State variables
        self.perception_latency_log = deque(maxlen=100)
        self.navigation_response_log = deque(maxlen=100)
        self.integration_success_log = deque(maxlen=1000)
        self.current_task = None
        self.task_start_time = None
        self.perception_active = False
        self.navigation_active = False

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detector/detections',
            self.detection_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.nav_status_sub = self.create_subscription(
            UInt8,  # Nav2 status
            '/navigation/status',
            self.nav_status_callback,
            10
        )

        # Publishers
        self.integration_score_pub = self.create_publisher(
            Float32,
            '/integration/validation_score',
            10
        )

        self.integration_status_pub = self.create_publisher(
            Bool,
            '/integration/validation_status',
            10
        )

        # Timers
        self.integration_timer = self.create_timer(0.1, self.integration_validation_callback)
        self.performance_timer = self.create_timer(1.0, self.performance_monitoring_callback)

        self.get_logger().info('Perception-navigation integration validator initialized')

    def detection_callback(self, msg):
        """Process detection messages."""
        detection_time = self.get_clock().now().nanoseconds * 1e-9

        # Log detection latency if we have corresponding tasks
        if self.current_task and self.task_start_time:
            latency = detection_time - self.task_start_time
            self.perception_latency_log.append(latency)

            # Check if detection is relevant to current task
            relevant_detection = self.is_detection_relevant(msg, self.current_task)
            if relevant_detection:
                self.handle_relevant_detection(msg, self.current_task)

    def odom_callback(self, msg):
        """Process odometry for navigation context."""
        if self.current_task and self.current_task['type'] == 'navigation':
            # Monitor navigation progress
            robot_pos = msg.pose.pose.position
            goal_pos = self.current_task.get('goal', [0, 0, 0])
            distance_to_goal = math.sqrt(
                (robot_pos.x - goal_pos[0])**2 +
                (robot_pos.y - goal_pos[1])**2
            )

            self.current_task['distance_to_goal'] = distance_to_goal

    def nav_status_callback(self, msg):
        """Process navigation status updates."""
        if msg.data == GoalStatus.STATUS_SUCCEEDED:
            self.handle_navigation_success()
        elif msg.data in [GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED]:
            self.handle_navigation_failure()

    def integration_validation_callback(self):
        """Main integration validation callback."""
        # Check for integration issues
        self.check_perception_navigation_sync()
        self.check_timing_constraints()
        self.evaluate_integration_performance()

    def check_perception_navigation_sync(self):
        """Check synchronization between perception and navigation."""
        # Verify that perception and navigation are properly coordinated
        if self.perception_active and self.navigation_active:
            # Check if perception outputs are being used by navigation
            self.evaluate_coordination()

    def check_timing_constraints(self):
        """Check timing constraints between perception and navigation."""
        # Verify that perception outputs arrive in time for navigation decisions
        if len(self.perception_latency_log) > 0:
            avg_latency = np.mean(list(self.perception_latency_log))
            max_acceptable_latency = 0.1  # 100ms for real-time navigation

            if avg_latency > max_acceptable_latency:
                self.get_logger().warn(
                    f'Perception latency too high: {avg_latency:.3f}s > {max_acceptable_latency:.3f}s')

    def evaluate_coordination(self):
        """Evaluate how well perception and navigation work together."""
        # This would check if detected objects influence navigation decisions
        # For example, if obstacles detected by perception cause navigation to replan
        pass

    def is_detection_relevant(self, detections, task):
        """Check if detections are relevant to current task."""
        if not task or 'roi' not in task:
            return False

        roi = task['roi']  # Region of interest
        for detection in detections.detections:
            bbox = detection.bbox
            center_x = bbox.center.x
            center_y = bbox.center.y

            if (roi[0] <= center_x <= roi[2] and
                roi[1] <= center_y <= roi[3]):
                return True

        return False

    def handle_relevant_detection(self, detections, task):
        """Handle relevant detection for current task."""
        # Update task with detection information
        if task['type'] == 'object_interaction':
            # Check if detected object matches target
            for detection in detections.detections:
                if detection.results and detection.results[0].id == task.get('target_object'):
                    task['object_detected'] = True
                    task['detection_confidence'] = detection.results[0].score

    def handle_navigation_success(self):
        """Handle navigation success event."""
        if self.current_task and self.current_task['type'] == 'navigation':
            execution_time = self.get_clock().now().nanoseconds * 1e-9 - self.task_start_time
            success = True

            self.integration_success_log.append({
                'task_id': self.current_task.get('id'),
                'success': success,
                'execution_time': execution_time,
                'timestamp': time.time()
            })

    def handle_navigation_failure(self):
        """Handle navigation failure event."""
        if self.current_task and self.current_task['type'] == 'navigation':
            execution_time = self.get_clock().now().nanoseconds * 1e-9 - self.task_start_time
            success = False

            self.integration_success_log.append({
                'task_id': self.current_task.get('id'),
                'success': success,
                'execution_time': execution_time,
                'failure_reason': 'navigation_failed',
                'timestamp': time.time()
            })

    def performance_monitoring_callback(self):
        """Monitor and publish performance metrics."""
        if len(self.integration_success_log) > 0:
            recent_results = list(self.integration_success_log)
            success_count = sum(1 for r in recent_results if r['success'])
            total_count = len(recent_results)

            success_rate = success_count / total_count if total_count > 0 else 0

            # Calculate integration score
            integration_score = self.calculate_integration_score(
                success_rate,
                list(self.perception_latency_log)
            )

            # Publish score
            score_msg = Float32()
            score_msg.data = integration_score
            self.integration_score_pub.publish(score_msg)

            # Publish status
            status_msg = Bool()
            status_msg.data = integration_score > 0.7  # Threshold for good integration
            self.integration_status_pub.publish(status_msg)

            self.get_logger().info(
                f'Integration Performance - Success Rate: {success_rate:.2f}, '
                f'Score: {integration_score:.2f}'
            )

    def calculate_integration_score(self, success_rate, latencies):
        """Calculate overall integration performance score."""
        if not latencies:
            avg_latency = 0.2  # Default to 200ms if no data
        else:
            avg_latency = np.mean(latencies)

        # Normalize components to 0-1 scale
        success_component = success_rate

        # Invert latency (lower is better)
        latency_component = max(0, min(1, (0.2 - avg_latency) / 0.2)) if avg_latency <= 0.2 else 0

        # Weighted combination
        integration_score = 0.6 * success_component + 0.4 * latency_component

        return integration_score

    def run_integration_test_suite(self, test_tasks):
        """Run a suite of integration tests."""
        results = {
            'total_tasks': len(test_tasks),
            'successful_tasks': 0,
            'average_integration_score': 0.0,
            'perception_latency_avg': 0.0,
            'navigation_success_rate': 0.0,
            'detailed_results': []
        }

        for i, task in enumerate(test_tasks):
            self.get_logger().info(f'Running integration test {i+1}/{len(test_tasks)}: {task["name"]}')

            # Set current task
            self.current_task = task
            self.task_start_time = self.get_clock().now().nanoseconds * 1e-9

            # Execute task (this would trigger the actual behavior)
            self.execute_task(task)

            # Wait for task completion
            timeout_time = time.time() + self.navigation_timeout
            while time.time() < timeout_time:
                rclpy.spin_once(self, timeout_sec=0.1)

                # Check if task is complete
                if self.is_task_complete(task):
                    break

            # Record results
            task_result = self.analyze_task_result(task)
            results['detailed_results'].append(task_result)

            if task_result['success']:
                results['successful_tasks'] += 1

        # Calculate aggregate metrics
        if results['detailed_results']:
            results['average_integration_score'] = np.mean([
                r['integration_score'] for r in results['detailed_results']
            ])
            results['perception_latency_avg'] = np.mean([
                r['avg_perception_latency'] for r in results['detailed_results'] if r['avg_perception_latency'] is not None
            ])
            results['navigation_success_rate'] = sum(
                1 for r in results['detailed_results'] if r.get('navigation_success', False)
            ) / len(results['detailed_results'])

        # Save results
        self.save_integration_results(results)

        self.get_logger().info(
            f'Integration Test Suite Complete:\n'
            f'  Success Rate: {results["successful_tasks"]}/{results["total_tasks"]} '
            f'({results["successful_tasks"]/results["total_tasks"]*100:.1f}%)\n'
            f'  Avg Integration Score: {results["average_integration_score"]:.2f}\n'
            f'  Avg Perception Latency: {results["perception_latency_avg"]:.3f}s\n'
            f'  Navigation Success Rate: {results["navigation_success_rate"]:.2f}'
        )

        return results

    def execute_task(self, task):
        """Execute a specific integration task."""
        task_type = task['type']

        if task_type == 'object_navigation':
            # Navigate to detect object
            self.execute_object_navigation_task(task)
        elif task_type == 'avoid_dynamic_obstacles':
            # Navigate while avoiding moving obstacles
            self.execute_dynamic_obstacle_task(task)
        elif task_type == 'follow_person':
            # Follow a detected person
            self.execute_person_following_task(task)

    def is_task_complete(self, task):
        """Check if task is complete."""
        # Implementation depends on task type
        if task['type'] == 'navigation':
            return hasattr(task, 'completed') and task['completed']
        return False

    def analyze_task_result(self, task):
        """Analyze results of completed task."""
        result = {
            'task_id': task.get('id'),
            'task_name': task.get('name'),
            'success': False,
            'integration_score': 0.0,
            'avg_perception_latency': None,
            'navigation_success': False,
            'execution_time': time.time() - self.task_start_time
        }

        # Determine success based on task completion
        if task.get('completed', False):
            result['success'] = True

            # Calculate integration-specific metrics
            if len(self.perception_latency_log) > 0:
                result['avg_perception_latency'] = np.mean(list(self.perception_latency_log))

            if task.get('navigation_completed', False):
                result['navigation_success'] = True

            # Calculate integration score
            result['integration_score'] = self.calculate_integration_score(
                1.0 if result['success'] else 0.0,
                list(self.perception_latency_log)
            )

        return result

    def save_integration_results(self, results):
        """Save integration validation results."""
        import json
        import os

        results_path = os.path.join(self.results_output_path, 'integration_validation_results.json')
        os.makedirs(self.results_output_path, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.get_logger().info(f'Integration results saved to {results_path}')


def main(args=None):
    rclpy.init(args=args)

    validator = PerceptionNavigationValidator()

    # Define test tasks
    test_tasks = [
        {
            'id': 'task_001',
            'name': 'Navigate to detect object',
            'type': 'object_navigation',
            'target_object': 'chair',
            'destination': [3.0, 2.0, 0.0],
            'roi': [0, 0, 640, 480]  # Full image ROI
        },
        {
            'id': 'task_002',
            'name': 'Avoid dynamic obstacles',
            'type': 'avoid_dynamic_obstacles',
            'destination': [5.0, 0.0, 0.0],
            'dynamic_obstacles': True
        },
        {
            'id': 'task_003',
            'name': 'Follow person',
            'type': 'follow_person',
            'follow_target': 'person',
            'duration': 30.0  # seconds
        }
    ]

    try:
        # Run integration test suite
        results = validator.run_integration_test_suite(test_tasks)

        # Continue monitoring
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 4. Validation Test Scripts

#### Automated Testing Script
```bash
#!/bin/bash
# validation_test_runner.sh

# Script to run automated validation tests for perception and navigation stacks

set -e  # Exit on any error

# Configuration
ROS_DISTRO=humble
WORKSPACE_DIR=~/ros2_ws
LOG_DIR=/tmp/validation_logs
DATE=$(date +"%Y%m%d_%H%M%S")

# Create log directory
mkdir -p $LOG_DIR/$DATE

echo "Starting automated validation tests..."

# Function to run perception validation
run_perception_validation() {
    echo "Running perception validation tests..."

    # Launch perception stack
    gnome-terminal -- bash -c "
        source /opt/ros/$ROS_DISTRO/setup.bash;
        source $WORKSPACE_DIR/install/setup.bash;
        ros2 launch perception_stack perception_launch.py
    " &
    PERCEPTION_PID=$!

    # Wait for perception stack to initialize
    sleep 10

    # Run validation node
    gnome-terminal -- bash -c "
        source /opt/ros/$ROS_DISTRO/setup.bash;
        source $WORKSPACE_DIR/install/setup.bash;
        ros2 run perception_validation object_detection_validator
    " &
    VALIDATION_PID=$!

    # Let tests run for specified time
    sleep 60

    # Kill processes
    kill $PERCEPTION_PID $VALIDATION_PID

    echo "Perception validation completed"
}

# Function to run navigation validation
run_navigation_validation() {
    echo "Running navigation validation tests..."

    # Launch navigation stack
    gnome-terminal -- bash -c "
        source /opt/ros/$ROS_DISTRO/setup.bash;
        source $WORKSPACE_DIR/install/setup.bash;
        ros2 launch navigation_stack navigation_launch.py
    " &
    NAVIGATION_PID=$!

    # Wait for navigation stack to initialize
    sleep 10

    # Run validation node
    gnome-terminal -- bash -c "
        source /opt/ros/$ROS_DISTRO/setup.bash;
        source $WORKSPACE_DIR/install/setup.bash;
        ros2 run navigation_validation navigation_stack_validator
    " &
    NAV_VALIDATION_PID=$!

    # Let tests run for specified time
    sleep 120

    # Kill processes
    kill $NAVIGATION_PID $NAV_VALIDATION_PID

    echo "Navigation validation completed"
}

# Function to run integration validation
run_integration_validation() {
    echo "Running integration validation tests..."

    # Launch both stacks
    gnome-terminal -- bash -c "
        source /opt/ros/$ROS_DISTRO/setup.bash;
        source $WORKSPACE_DIR/install/setup.bash;
        ros2 launch perception_stack perception_launch.py
    " &
    PERCEPTION_PID=$!

    gnome-terminal -- bash -c "
        source /opt/ros/$ROS_DISTRO/setup.bash;
        source $WORKSPACE_DIR/install/setup.bash;
        ros2 launch navigation_stack navigation_launch.py
    " &
    NAVIGATION_PID=$!

    # Wait for stacks to initialize
    sleep 15

    # Run integration validation
    gnome-terminal -- bash -c "
        source /opt/ros/$ROS_DISTRO/setup.bash;
        source $WORKSPACE_DIR/install/setup.bash;
        ros2 run integration_validation perception_navigation_validator
    " &
    INTEGRATION_PID=$!

    # Let tests run for specified time
    sleep 180

    # Kill processes
    kill $PERCEPTION_PID $NAVIGATION_PID $INTEGRATION_PID

    echo "Integration validation completed"
}

# Function to run stress tests
run_stress_tests() {
    echo "Running stress tests..."

    # Run multiple scenarios in parallel
    for i in {1..5}; do
        echo "Running stress test iteration $i"

        # Launch scenario
        gnome-terminal -- bash -c "
            source /opt/ros/$ROS_DISTRO/setup.bash;
            source $WORKSPACE_DIR/install/setup.bash;
            ros2 launch validation_scenarios scenario_$i.launch.py
        " &
        SCENARIO_PID=$!

        sleep 30

        kill $SCENARIO_PID
        sleep 5
    done

    echo "Stress tests completed"
}

# Main execution
echo "Starting validation test suite at $DATE"

# Run each validation component
run_perception_validation
sleep 5

run_navigation_validation
sleep 5

run_integration_validation
sleep 5

run_stress_tests

# Generate summary report
echo "Generating validation summary report..."
cat > $LOG_DIR/$DATE/summary_report.txt << EOF
Validation Test Summary - $DATE

1. Perception Validation
   - Status: COMPLETED
   - Duration: 60 seconds
   - Metrics: Precision, Recall, mAP, Inference Time

2. Navigation Validation
   - Status: COMPLETED
   - Duration: 120 seconds
   - Metrics: Success Rate, Path Efficiency, Stability

3. Integration Validation
   - Status: COMPLETED
   - Duration: 180 seconds
   - Metrics: Integration Score, Coordination, Latency

4. Stress Tests
   - Status: COMPLETED
   - Iterations: 5
   - Scenarios: Various environments and conditions

Overall System Performance:
- Perception: [TO BE FILLED BY VALIDATION NODES]
- Navigation: [TO BE FILLED BY VALIDATION NODES]
- Integration: [TO BE FILLED BY VALIDATION NODES]

Recommendations:
- [TO BE GENERATED BY ANALYSIS SCRIPTS]
EOF

echo "Validation test suite completed. Results saved to $LOG_DIR/$DATE"