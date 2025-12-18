---
sidebar_position: 33
---

# VSLAM Implementation with RealSense Data

## Learning Objectives
By the end of this module, students will be able to:
- Implement Visual Simultaneous Localization and Mapping (VSLAM) systems using RealSense camera data
- Configure and calibrate RealSense D435i for VSLAM applications
- Integrate ORB-SLAM3 with ROS 2 for real-time mapping and localization
- Optimize VSLAM performance on RTX workstations with GPU acceleration
- Evaluate VSLAM accuracy and reliability in indoor environments
- Integrate VSLAM with robot navigation systems

## Theory

### Visual SLAM Fundamentals

Visual SLAM (Simultaneous Localization and Mapping) is a critical capability for autonomous robots, allowing them to build a map of unknown environments while simultaneously localizing themselves within that map using visual data.

### Key Components of VSLAM Systems

#### 1. Visual Odometry
- Feature detection and matching
- Motion estimation from visual correspondences
- Local map refinement

#### 2. Loop Closure Detection
- Identifying previously visited locations
- Correcting accumulated drift
- Maintaining global map consistency

#### 3. Map Representation
- Keyframe-based maps
- Point cloud representations
- Graph optimization

### ORB-SLAM3 Architecture

ORB-SLAM3 is a state-of-the-art VSLAM system that supports:
- Monocular, stereo, and RGB-D cameras
- Multi-map management for relocalization
- Place recognition and loop closure
- Bundle adjustment for map optimization

### RealSense D435i for VSLAM

The Intel RealSense D435i provides:
- RGB camera for visual features
- Depth sensor for 3D reconstruction
- IMU for motion prediction
- Hardware synchronization capabilities

### SLAM vs. VSLAM vs. VO
- **Visual Odometry (VO)**: Estimates motion from visual features (local)
- **Visual SLAM (VSLAM)**: VO + mapping + loop closure (global)
- **LiDAR SLAM**: Uses LiDAR instead of visual data

## Implementation

### Prerequisites
- Intel RealSense D435i camera
- Ubuntu 22.04 with ROS 2 Humble
- OpenCV and Pangolin dependencies
- ORB-SLAM3 installed
- RTX GPU for accelerated processing (optional but recommended)

### RealSense D435i Setup

#### 1. Hardware Configuration
```bash
# Install RealSense SDK
sudo apt install librealsense2-dev librealsense2-utils

# Install ROS 2 RealSense package
sudo apt install ros-humble-realsense2-camera

# Check camera connection
rs-enumerate-devices
```

#### 2. Camera Calibration
```bash
# Launch RealSense camera with calibration
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=camera \
  enable_infra1:=false \
  enable_infra2:=false \
  enable_color:=true \
  enable_depth:=true \
  enable_gyro:=true \
  enable_accel:=true \
  unite_imu_method:=linear_interpolation
```

#### 3. Camera Calibration Files
Create `camera_calibration.py` for intrinsic and extrinsic calibration:

```python
#!/usr/bin/env python3
"""
RealSense D435i camera calibration script.
"""
import cv2
import numpy as np
import yaml
import os
from pathlib import Path

class RealSenseCalibrator:
    """Class to calibrate RealSense D435i camera parameters."""

    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        """
        Initialize calibrator with checkerboard parameters.

        Args:
            checkerboard_size: Tuple of (corners_x, corners_y) in checkerboard
            square_size: Size of each square in meters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size

        # Arrays to store object points and image points
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane

        # Prepare object points [[0,0,0], [0.025,0,0], [0.05,0,0], ...]
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    def find_checkerboard(self, img):
        """
        Find checkerboard corners in image.

        Args:
            img: Input image

        Returns:
            corners: Corner coordinates if found, None otherwise
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            return corners_refined

        return None

    def calibrate_camera(self, images_path, output_path):
        """
        Calibrate camera using checkerboard images.

        Args:
            images_path: Path to directory containing checkerboard images
            output_path: Path to save calibration file
        """
        image_files = list(Path(images_path).glob("*.jpg")) + list(Path(images_path).glob("*.png"))

        if not image_files:
            raise ValueError(f"No images found in {images_path}")

        valid_images = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))

            corners = self.find_checkerboard(img)

            if corners is not None:
                # Add object points and image points
                self.obj_points.append(self.objp)
                self.img_points.append(corners)
                valid_images += 1

                # Draw and display corners
                cv2.drawChessboardCorners(img, self.checkerboard_size, corners, True)
                cv2.imshow('Calibration', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        if valid_images < 10:
            raise ValueError(f"Not enough valid images for calibration. Found {valid_images}, need at least 10.")

        print(f"Found {valid_images} valid images for calibration")

        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            (img.shape[1], img.shape[0]),
            None,
            None
        )

        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.obj_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.obj_points[i],
                rvecs[i],
                tvecs[i],
                mtx,
                dist
            )
            error = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        avg_error = total_error / len(self.obj_points)

        print(f"Calibration completed!")
        print(f"Reprojection error: {avg_error}")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients: {dist.flatten()}")

        # Save calibration data
        self.save_calibration(mtx, dist, avg_error, output_path)

        return mtx, dist, avg_error

    def save_calibration(self, camera_matrix, dist_coeffs, error, output_path):
        """
        Save calibration data in ROS-compatible format.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3, ...]
            error: Average reprojection error
            output_path: Path to save calibration file
        """
        calibration_data = {
            'image_width': 640,  # Update based on your camera resolution
            'image_height': 480,
            'camera_name': 'camera',
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': camera_matrix.flatten().tolist()
            },
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': dist_coeffs.flatten()[:5].tolist()  # Take first 5 coefficients
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [1, 0, 0, 0, 1, 0, 0, 0, 1]
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': [
                    camera_matrix[0,0], 0, camera_matrix[0,2], 0,
                    0, camera_matrix[1,1], camera_matrix[1,2], 0,
                    0, 0, 1, 0
                ]
            },
            'calibration_error': float(error)
        }

        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)

        print(f"Calibration saved to {output_path}")

# Usage example
if __name__ == "__main__":
    calibrator = RealSenseCalibrator(checkerboard_size=(9, 6), square_size=0.025)

    # Calibrate using checkerboard images
    # calibrator.calibrate_camera("./calibration_images", "./camera_calib.yaml")
```

### ORB-SLAM3 Installation and Configuration

#### 1. Dependencies Installation
```bash
# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake pkg-config libatlas-base-dev libeigen3-dev libsuitesparse-dev
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libglew-dev libglfw3-dev libboost-all-dev

# Install Pangolin for visualization
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake .. -DCPP11_NO_BOOST=1
make -j$(nproc)
sudo make install

# Install ORB-SLAM3
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```

#### 2. ORB-SLAM3 ROS 2 Wrapper
Create `orb_slam3_ros_wrapper.cpp`:

```cpp
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "System.h"

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber() = default;
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

public:
    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber* pImuGb, const string& strCameraTopic);
    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

public:
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber* mpImuGb;
    string mStrCameraTopic;
    queue<sensor_msgs::ImageConstPtr> img0Buf;
    std::mutex mBufMutex;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD_Inertial_Oct22");
    ros::NodeHandle nh;

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3_RGBD_Inertial path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO,true);

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM, &imugb, "/camera/color/image_raw");
    cout << endl << "-------" << endl;
    cout << "Starting the ROS node" << endl;

    ros::Subscriber sub_imu = nh.subscribe("/camera/accel/sample", 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img0 = nh.subscribe(igb.mStrCameraTopic, 1000, &ImageGrabber::GrabImage, &igb);

    // Need to set a callback for the imu queue to be processed
    ros::Timer timer = nh.createTimer(ros::Duration(0.001), boost::bind(&ImageGrabber::SyncWithImu,&igb));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM.txt");
    SLAM.SaveTrajectoryTUM("FrameTrajectory_TUM.txt");
    SLAM.SaveTrajectoryKITTI("FrameTrajectory_KITTI.txt");

    ros::shutdown();

    return 0;
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
}

ImageGrabber::ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber* pImuGb, const string& strCameraTopic):
       mpSLAM(pSLAM), mpImuGb(pImuGb), mStrCameraTopic(strCameraTopic)
{
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    mBufMutex.lock();
    if (!img0Buf.empty())
        img0Buf.pop();
    img0Buf.push(msg);
    mBufMutex.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    if(cv_ptr->image.type()==CV_32FC1)
    {
        cv::Mat mat = cv_ptr->image.clone();
        return mat;
    }
    else
    {
        std::cout << "Error type" << std::endl;
        return cv_ptr->image.clone();
    }
}

void ImageGrabber::SyncWithImu()
{
    mBufMutex.lock();
    if(!img0Buf.empty() && !mpImuGb->imuBuf.empty())
    {
        // Last image, for instance
        const sensor_msgs::ImageConstPtr &img0_msg = img0Buf.front();

        // Intermediate variables
        vector<sensor_msgs::ImuConstPtr> IMUs;
        std::cout << "Imu buffer size: " << mpImuGb->imuBuf.size() << std::endl;

        // Get the last IMU in the buffer
        const sensor_msgs::ImuConstPtr &last_imu = mpImuGb->imuBuf.back();

        // Get all the IMU messages in the buffer between the first image and the last IMU
        while((!mpImuGb->imuBuf.empty()) && (mpImuGb->imuBuf.front()->header.stamp.toSec() < img0_msg->header.stamp.toSec()))
        {
            // Store the IMU message
            IMUs.push_back(mpImuGb->imuBuf.front());
            mpImuGb->imuBuf.pop();
        }

        // Get the image
        cv::Mat im = GetImage(img0_msg);

        // Set the image and IMU to the SLAM system
        mpSLAM->TrackRGBDInertial(im, im, img0_msg->header.stamp.toSec(), IMUs);

        // Remove the processed image
        img0Buf.pop();
    }
    mBufMutex.unlock();
}
```

#### 3. RealSense Integration Configuration
Create `realsense_orb_slam3.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    camera_namespace = LaunchConfiguration('camera_namespace', default='camera')
    vocab_path = LaunchConfiguration('vocab_path', default='/path/to/Vocabulary/ORBvoc.txt')
    settings_path = LaunchConfiguration('settings_path', default='/path/to/RealSense_D435I.yaml')

    # RealSense camera launch
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'camera_namespace': camera_namespace,
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': 'linear_interpolation',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'false',
            'initial_reset': 'true'
        }.items()
    )

    # ORB-SLAM3 node
    orb_slam3_node = Node(
        package='orb_slam3_ros',
        executable='orb_slam3_rgbd_inertial_node',
        name='orb_slam3',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'vocab_path': vocab_path},
            {'settings_path': settings_path},
            {'camera_topic': [camera_namespace, '/color/image_raw']},
            {'depth_topic': [camera_namespace, '/aligned_depth_to_color/image_raw']},
            {'imu_topic': [camera_namespace, '/accel/sample']}
        ],
        output='screen'
    )

    # TF broadcasters
    tf_broadcaster = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'camera_link']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'vocab_path',
            default_value='/home/user/ORBSLAM3/Vocabulary/ORBvoc.txt',
            description='Path to ORB vocabulary file'
        ),
        DeclareLaunchArgument(
            'settings_path',
            default_value='/home/user/ORBSLAM3/Examples/ROS/ORB_SLAM3_ROS/Config/RealSense_D435I.yaml',
            description='Path to ORB-SLAM3 settings file'
        ),
        realsense_launch,
        orb_slam3_node,
        tf_broadcaster
    ])
```

#### 4. ORB-SLAM3 Configuration File
Create `RealSense_D435I.yaml`:

```yaml
%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV)
Camera.fx: 615.863
Camera.fy: 615.829
Camera.cx: 323.101
Camera.cy: 237.094

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
Camera.k3: 0.0

Camera.width: 640
Camera.height: 480

# Camera frames per second
Camera.fps: 30.0

# IR projector baseline times fx (aprox.)
Camera.bf: 40.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# Stereo Parameters
#--------------------------------------------------------------------------------------------
# The stereo baseline is taken from the calibration of the IMU and the left and right cameras
# The IMU is assumed to be in the same plane of the left and right cameras
# The IMU is assumed to be in the middle of the left and right cameras
# The ORBextractor is not used in the stereo-inertial case, therefore its parameters are not included

# The scaling factor from the stereo camera is taken from the calibration of the stereo camera
# The ORBextractor is not used in the stereo-inertial case, therefore its parameters are not included
# The ORBextractor is not used in the stereo-inertial case, therefore its parameters are not included

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1200

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
```

### GPU-Accelerated VSLAM Implementation

#### 1. CUDA-Optimized Feature Extraction
```cpp
// cuda_feature_extractor.cu
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

class CUDASLAMFeatureExtractor {
public:
    CUDASLAMFeatureExtractor(int width, int height);
    ~CUDASLAMFeatureExtractor();

    void extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void initializeGPU();

private:
    cv::Ptr<cv::cuda::ORB> cuda_orb;
    cv::cuda::GpuMat gpu_image;
    cv::cuda::GpuMat gpu_gray;
    cv::cuda::GpuMat gpu_keypoints;
    cv::cuda::GpuMat gpu_descriptors;

    int img_width, img_height;
};

CUDASLAMFeatureExtractor::CUDASLAMFeatureExtractor(int width, int height)
    : img_width(width), img_height(height) {

    // Initialize CUDA ORB detector
    cuda_orb = cv::cuda::ORB::create(2000); // 2000 features
    cuda_orb->setMaxFeatures(2000);
    cuda_orb->setScaleFactor(1.2f);
    cuda_orb->setNLevels(8);
    cuda_orb->setEdgeThreshold(31);
    cuda_orb->setFirstLevel(0);
    cuda_orb->setWTA_K(2);
    cuda_orb->setScoreType(cv::ORB::HARRIS_SCORE);
    cuda_orb->setPatchSize(31);
    cuda_orb->setFastThreshold(20);

    // Allocate GPU memory
    gpu_image = cv::cuda::GpuMat(img_height, img_width, CV_8UC3);
    gpu_gray = cv::cuda::GpuMat(img_height, img_width, CV_8UC1);
}

void CUDASLAMFeatureExtractor::extractFeatures(const cv::Mat& image,
                                               std::vector<cv::KeyPoint>& keypoints,
                                               cv::Mat& descriptors) {
    // Upload image to GPU
    gpu_image.upload(image);

    // Convert to grayscale on GPU
    cv::cuda::cvtColor(gpu_image, gpu_gray, cv::COLOR_BGR2GRAY);

    // Detect and compute features on GPU
    std::vector<cv::KeyPoint> temp_keypoints;
    cv::cuda::GpuMat temp_descriptors;

    cuda_orb->detectAndCompute(gpu_gray, cv::cuda::GpuMat(), temp_keypoints, temp_descriptors);

    // Download results
    if (!temp_descriptors.empty()) {
        temp_descriptors.download(descriptors);
    }

    keypoints = temp_keypoints;
}

// Alternative implementation with custom CUDA kernels for feature matching
__global__ void compute_descriptor_distances(float* descriptors1, float* descriptors2,
                                           float* distances, int num_desc1, int num_desc2, int desc_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_desc1 * num_desc2) {
        int desc1_idx = idx / num_desc2;
        int desc2_idx = idx % num_desc2;

        float dist = 0.0f;
        for (int i = 0; i < desc_size; i++) {
            float diff = descriptors1[desc1_idx * desc_size + i] -
                         descriptors2[desc2_idx * desc_size + i];
            dist += diff * diff;
        }

        distances[idx] = sqrtf(dist);
    }
}
```

#### 2. RTAB-Map Integration for Dense Mapping
```python
# rtabmap_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as transforms

class RTABMapIntegrator(Node):
    """
    Integrates RTAB-Map with RealSense for dense 3D mapping.
    """

    def __init__(self):
        super().__init__('rtabmap_integrator')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Create publishers
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/rtabmap/grid_map',
            10
        )

        # RTAB-Map parameters
        self.rtabmap_params = {
            'Mem/RehearsalSimilarity': '0.45',
            'Mem/NotLinkedNodesKept': 'false',
            'Kp/DetectorStrategy': '6',  # SURF
            'SURF/HessianThreshold': '500',
            'GFTT/MinDistance': '10',
            'GFTT/QualityLevel': '0.001',
            'RGBD/ProximityPathMaxNeighbors': '10',
            'RGBD/ProximityPathFilteringRadius': '0.1',
            'RGBD/OptimizeFromGraphEnd': 'false',
            'Optimizer/Slam2D': 'true',
            'Reg/Force3DoF': 'true',
            'Grid/FromDepth': 'true',
            'Grid/CellSize': '0.05',
            'Grid/RangeMax': '5.0',
        }

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Frame buffers
        self.rgb_buffer = []
        self.depth_buffer = []

        # Processing timer
        self.process_timer = self.create_timer(0.1, self.process_frames)

        self.get_logger().info('RTAB-Map integrator initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)

    def rgb_callback(self, msg):
        """Process RGB image from RealSense."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_buffer.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, cv_image))

            # Keep only recent frames
            self.rgb_buffer = [f for f in self.rgb_buffer if
                              (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - f[0] < 5.0]
        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        """Process depth image from RealSense."""
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_buffer.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, cv_depth))

            # Keep only recent frames
            self.depth_buffer = [f for f in self.depth_buffer if
                                (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - f[0] < 5.0]
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def process_frames(self):
        """Process synchronized RGB-D frames for mapping."""
        if not self.rgb_buffer or not self.depth_buffer or self.camera_matrix is None:
            return

        # Find closest matching frames
        best_match = self.find_closest_frames()
        if best_match is None:
            return

        rgb_stamp, rgb_img = best_match[0]
        depth_stamp, depth_img = best_match[1]

        # Process frame for RTAB-Map
        try:
            self.process_rtabmap_frame(rgb_img, depth_img)
        except Exception as e:
            self.get_logger().error(f'Error in RTAB-Map processing: {e}')

    def find_closest_frames(self):
        """Find temporally closest RGB and depth frames."""
        min_diff = float('inf')
        best_match = None

        for rgb_time, rgb_img in self.rgb_buffer:
            for depth_time, depth_img in self.depth_buffer:
                time_diff = abs(rgb_time - depth_time)
                if time_diff < min_diff and time_diff < 0.1:  # Less than 100ms apart
                    min_diff = time_diff
                    best_match = ((rgb_time, rgb_img), (depth_time, depth_img))

        return best_match

    def process_rtabmap_frame(self, rgb_img, depth_img):
        """Process a synchronized RGB-D frame for RTAB-Map."""
        # In a real implementation, you would interface with RTAB-Map
        # through its ROS interface or C++ API
        # For this example, we'll simulate the process

        # Convert depth to float and scale
        depth_float = depth_img.astype(np.float32) / 1000.0  # Convert mm to meters

        # Create point cloud from RGB-D
        points = self.create_point_cloud(rgb_img, depth_float, self.camera_matrix)

        # In real implementation, this would be sent to RTAB-Map
        # self.send_to_rtabmap(rgb_img, depth_float, points)

        # Log processing info
        valid_pixels = np.count_nonzero(~np.isnan(depth_float.flatten()))
        self.get_logger().info(f'Processed frame: {valid_pixels} valid depth pixels')

    def create_point_cloud(self, rgb_img, depth_img, camera_matrix):
        """Create point cloud from RGB-D data."""
        height, width = depth_img.shape

        # Create coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Convert pixel coordinates to camera coordinates
        x_cam = (u_coords - camera_matrix[0, 2]) * depth_img / camera_matrix[0, 0]
        y_cam = (v_coords - camera_matrix[1, 2]) * depth_img / camera_matrix[1, 1]

        # Create 3D points
        valid_mask = (depth_img > 0) & (depth_img < 10.0)  # Valid depth range
        points_3d = np.stack([x_cam[valid_mask], y_cam[valid_mask], depth_img[valid_mask]], axis=-1)

        # Get corresponding colors
        colors = rgb_img[valid_mask] if rgb_img is not None else None

        return points_3d, colors

def main(args=None):
    rclpy.init(args=args)

    node = RTABMapIntegrator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Performance Optimization and Evaluation

#### 1. VSLAM Performance Monitoring
```python
# vslam_performance_monitor.py
import rospy
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
import time
import matplotlib.pyplot as plt
from collections import deque

class VSLAMPerformanceMonitor:
    """
    Monitors VSLAM performance metrics in real-time.
    """

    def __init__(self):
        rospy.init_node('vslam_performance_monitor')

        # Performance metrics
        self.processing_times = deque(maxlen=100)
        self.tracking_rates = deque(maxlen=100)
        self.localization_errors = deque(maxlen=100)
        self.map_consistency_scores = deque(maxlen=100)

        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.pose_sub = rospy.Subscriber('/orb_slam3/camera_pose', PoseStamped, self.pose_callback)

        # Publishers
        self.timing_pub = rospy.Publisher('/vslam/timing_stats', Float32, queue_size=10)
        self.rate_pub = rospy.Publisher('/vslam/tracking_rate', Float32, queue_size=10)
        self.error_pub = rospy.Publisher('/vslam/localization_error', Float32, queue_size=10)

        # Visualization
        self.marker_pub = rospy.Publisher('/vslam/performance_markers', MarkerArray, queue_size=10)

        # Timing
        self.last_image_time = None
        self.last_pose_time = None

        # Performance thresholds
        self.config = {
            'max_processing_time': 0.1,  # 100ms
            'min_tracking_rate': 10.0,   # 10Hz
            'max_localization_error': 0.1 # 10cm
        }

        # Start monitoring
        self.monitor_timer = rospy.Timer(rospy.Duration(1.0), self.publish_performance_stats)

        rospy.loginfo('VSLAM Performance Monitor initialized')

    def image_callback(self, msg):
        """Process image timing."""
        current_time = rospy.Time.now().to_sec()

        if self.last_image_time is not None:
            processing_time = current_time - self.last_image_time
            self.processing_times.append(processing_time)

        self.last_image_time = current_time

    def pose_callback(self, msg):
        """Process pose estimation."""
        current_time = rospy.Time.now().to_sec()

        if self.last_pose_time is not None:
            time_diff = current_time - self.last_pose_time
            tracking_rate = 1.0 / time_diff if time_diff > 0 else 0.0
            self.tracking_rates.append(tracking_rate)

        self.last_pose_time = current_time

        # Calculate localization accuracy (would compare with ground truth in real scenario)
        # For simulation, we'll simulate error based on tracking confidence
        simulated_error = np.random.normal(0.02, 0.01)  # 2cm average error
        self.localization_errors.append(abs(simulated_error))

    def publish_performance_stats(self, event):
        """Publish performance statistics."""
        if len(self.processing_times) > 0:
            avg_processing_time = np.mean(self.processing_times)
            self.timing_pub.publish(Float32(avg_processing_time))

            # Check if performance is degrading
            if avg_processing_time > self.config['max_processing_time']:
                rospy.logwarn(f'VSLAM processing time too high: {avg_processing_time:.3f}s')

        if len(self.tracking_rates) > 0:
            avg_tracking_rate = np.mean(self.tracking_rates)
            self.rate_pub.publish(Float32(avg_tracking_rate))

            if avg_tracking_rate < self.config['min_tracking_rate']:
                rospy.logwarn(f'VSLAM tracking rate too low: {avg_tracking_rate:.2f}Hz')

        if len(self.localization_errors) > 0:
            avg_localization_error = np.mean(self.localization_errors)
            self.error_pub.publish(Float32(avg_localization_error))

            if avg_localization_error > self.config['max_localization_error']:
                rospy.logwarn(f'VSLAM localization error too high: {avg_localization_error:.3f}m')

        # Visualize performance metrics
        self.visualize_performance()

    def visualize_performance(self):
        """Visualize performance metrics in RViz."""
        marker_array = MarkerArray()

        # Create performance visualization markers
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "vslam_performance"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # Set position to current robot position (would come from pose topic in real implementation)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0

        # Set text with performance stats
        perf_text = f"VSLAM Performance:\n"
        if len(self.processing_times) > 0:
            perf_text += f"Processing: {np.mean(self.processing_times)*1000:.1f}ms\n"
        if len(self.tracking_rates) > 0:
            perf_text += f"Tracking: {np.mean(self.tracking_rates):.1f}Hz\n"
        if len(self.localization_errors) > 0:
            perf_text += f"Accuracy: {np.mean(self.localization_errors)*100:.1f}cm"

        marker.text = perf_text
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker_array.markers.append(marker)

        # Publish markers
        self.marker_pub.publish(marker_array)

    def generate_performance_report(self):
        """Generate detailed performance report."""
        if len(self.processing_times) == 0:
            rospy.loginfo('Insufficient data for performance report')
            return

        report = {
            'timestamp': rospy.Time.now().to_sec(),
            'processing_stats': {
                'mean_time_ms': np.mean(self.processing_times) * 1000,
                'std_time_ms': np.std(self.processing_times) * 1000,
                'min_time_ms': np.min(self.processing_times) * 1000,
                'max_time_ms': np.max(self.processing_times) * 1000,
                'percentile_95_ms': np.percentile(self.processing_times, 95) * 1000
            },
            'tracking_stats': {
                'mean_rate_hz': np.mean(self.tracking_rates) if len(self.tracking_rates) > 0 else 0,
                'std_rate_hz': np.std(self.tracking_rates) if len(self.tracking_rates) > 0 else 0,
                'min_rate_hz': np.min(self.tracking_rates) if len(self.tracking_rates) > 0 else 0,
                'max_rate_hz': np.max(self.tracking_rates) if len(self.tracking_rates) > 0 else 0
            },
            'accuracy_stats': {
                'mean_error_cm': np.mean(self.localization_errors) * 100 if len(self.localization_errors) > 0 else 0,
                'std_error_cm': np.std(self.localization_errors) * 100 if len(self.localization_errors) > 0 else 0,
                'max_error_cm': np.max(self.localization_errors) * 100 if len(self.localization_errors) > 0 else 0
            }
        }

        rospy.loginfo(f'VSLAM Performance Report:\n{report}')
        return report

def main():
    monitor = VSLAMPerformanceMonitor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down VSLAM Performance Monitor')
        report = monitor.generate_performance_report()
        # Would save report to file in real implementation

if __name__ == '__main__':
    main()
```

### Integration with Navigation Stack

#### 1. VSLAM-Nav2 Bridge
```python
# vslam_nav2_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import numpy as np
import tf2_ros

class VSLAMNav2Bridge(Node):
    """
    Bridge between VSLAM system and ROS 2 Navigation stack.
    Converts VSLAM pose estimates to navigation-compatible format.
    """

    def __init__(self):
        super().__init__('vslam_nav2_bridge')

        # Create subscribers
        self.vslam_pose_sub = self.create_subscription(
            PoseStamped,
            '/orb_slam3/camera_pose',
            self.vslam_pose_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Create publishers
        self.amcl_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            '/vslam/odom',
            10
        )

        self.map_to_odom_pub = self.create_publisher(
            TransformStamped,
            '/vslam/map_to_odom',
            10
        )

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # State variables
        self.vslam_pose = None
        self.odom_pose = None
        self.initial_alignment_performed = False

        # Covariance parameters for AMCL
        self.pose_covariance = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.1])  # Diagonal covariance matrix

        # Parameters
        self.use_vslam_for_global_loc = self.declare_parameter(
            'use_vslam_for_global_localization',
            True
        ).get_parameter_value().bool_value

        self.vslam_confidence_threshold = self.declare_parameter(
            'vslam_confidence_threshold',
            0.7
        ).get_parameter_value().double_value

        self.get_logger().info('VSLAM-Nav2 bridge initialized')

    def vslam_pose_callback(self, msg):
        """Process VSLAM pose estimate."""
        self.vslam_pose = msg

        # Convert VSLAM pose to odom frame if needed
        if self.odom_pose is not None:
            self.publish_vslam_odom()

        # Check if we should use VSLAM for global localization
        if self.use_vslam_for_global_loc and not self.initial_alignment_performed:
            self.perform_initial_alignment()

    def odom_callback(self, msg):
        """Process odometry data."""
        self.odom_pose = msg

    def publish_vslam_odom(self):
        """Publish VSLAM-based odometry for navigation."""
        if self.vslam_pose is None:
            return

        # Create odometry message from VSLAM pose
        odom_msg = Odometry()
        odom_msg.header = self.vslam_pose.header
        odom_msg.header.frame_id = 'odom'  # Nav2 expects odom frame
        odom_msg.child_frame_id = 'base_footprint'

        # Copy position and orientation
        odom_msg.pose.pose = self.vslam_pose.pose

        # Set covariance based on VSLAM confidence
        confidence = self.estimate_vslam_confidence()
        scaled_covariance = self.pose_covariance * (1.0 / max(confidence, 0.1))
        odom_msg.pose.covariance = scaled_covariance.flatten().tolist()

        # For velocity, we might estimate from pose differences
        # In this example, we'll set it to zero
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'

        t.transform.translation.x = self.vslam_pose.pose.position.x
        t.transform.translation.y = self.vslam_pose.pose.position.y
        t.transform.translation.z = self.vslam_pose.pose.position.z

        t.transform.rotation = self.vslam_pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)

    def estimate_vslam_confidence(self):
        """Estimate confidence in VSLAM pose based on tracking quality."""
        # In a real implementation, this would use VSLAM's internal tracking quality metrics
        # For this example, we'll simulate confidence based on time since last update
        if hasattr(self, '_last_vslam_time'):
            time_since_update = self.get_clock().now().nanoseconds - self._last_vslam_time
            # Decrease confidence over time if no updates
            time_decay = min(time_since_update * 1e-9 * 0.1, 0.5)  # 0.5 max decay per second
            confidence = max(0.5, 1.0 - time_decay)  # Base confidence of 0.5, decaying
        else:
            confidence = 1.0  # Initial high confidence

        self._last_vslam_time = self.get_clock().now().nanoseconds
        return confidence

    def perform_initial_alignment(self):
        """Perform initial alignment between VSLAM and navigation frames."""
        if self.vslam_pose is None or self.odom_pose is None:
            return

        # Publish initial pose estimate to AMCL for global localization
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.frame_id = 'map'
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()

        # Use VSLAM pose as initial estimate
        initial_pose_msg.pose.pose = self.vslam_pose.pose
        initial_pose_msg.pose.covariance = self.pose_covariance.flatten().tolist()

        # Publish with appropriate confidence
        if self.estimate_vslam_confidence() > self.vslam_confidence_threshold:
            self.amcl_pose_pub.publish(initial_pose_msg)
            self.initial_alignment_performed = True
            self.get_logger().info('Initial alignment with VSLAM performed')
        else:
            self.get_logger().warn('VSLAM confidence too low for initial alignment')

    def get_vslam_map(self):
        """Retrieve map from VSLAM system for Nav2 costmap initialization."""
        # In a real implementation, this would interface with VSLAM's map representation
        # For now, we'll return a placeholder
        pass

def main(args=None):
    rclpy.init(args=args)

    bridge = VSLAMNav2Bridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises

1. Set up ORB-SLAM3 with RealSense D435i on your robot platform
2. Calibrate your RealSense camera using the provided calibration tools
3. Implement GPU-accelerated feature extraction for improved performance
4. Create a dense 3D map of your apartment environment using RTAB-Map
5. Integrate VSLAM with your navigation stack for global localization
6. Evaluate VSLAM performance in different lighting conditions
7. Optimize VSLAM parameters for your specific environment
8. Test VSLAM robustness to dynamic objects and changing scenes

## References

1. ORB-SLAM3 GitHub: https://github.com/UZ-SLAMLab/ORB_SLAM3
2. Intel RealSense Documentation: https://www.intelrealsense.com/sdk-2/
3. RTAB-Map ROS Package: http://wiki.ros.org/rtabmap_ros
4. Visual SLAM Survey: https://arxiv.org/abs/1606.05830
5. Real-Time Dense Mapping: https://arxiv.org/abs/1804.06510

## Further Reading

- Advanced VSLAM techniques: Direct methods, semi-direct methods
- Deep learning integration with traditional VSLAM
- Multi-session mapping and lifelong SLAM
- SLAM in dynamic environments with moving objects
- Evaluation metrics and benchmarking for SLAM systems
- Sensor fusion with IMU and LiDAR for robust SLAM