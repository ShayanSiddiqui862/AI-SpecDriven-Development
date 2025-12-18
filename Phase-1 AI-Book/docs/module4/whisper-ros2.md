---
sidebar_position: 40
---

# Whisper Integration for Voice Command Processing in ROS 2

## Learning Objectives
By the end of this module, students will be able to:
- Integrate OpenAI Whisper with ROS 2 for real-time voice command processing
- Configure Whisper models for different performance and accuracy requirements
- Implement voice activity detection and noise filtering for robotics applications
- Create ROS 2 nodes that process voice commands and generate appropriate responses
- Optimize Whisper inference for real-time performance on robotics hardware

## Theory

### Speech Recognition in Robotics

Speech recognition is a critical component of human-robot interaction, enabling natural communication between humans and robots. In robotics applications, speech recognition systems must handle:

- **Real-time processing**: Commands must be processed with minimal latency
- **Noisy environments**: Robustness to environmental sounds and robot-generated noise
- **Limited vocabulary**: Focus on command-specific recognition rather than general speech
- **Low-power operation**: Efficient processing on embedded systems

### Whisper Architecture

OpenAI's Whisper is a transformer-based speech recognition model that excels at:

- **Multilingual support**: Can recognize speech in multiple languages
- **Robustness**: Performs well in noisy environments
- **Zero-shot capabilities**: Works without fine-tuning for specific domains
- **Timestamp alignment**: Provides word-level timing information

### Key Concepts

#### 1. Automatic Speech Recognition (ASR)
- **Acoustic Model**: Maps audio features to phonemes
- **Language Model**: Determines likely word sequences
- **Decoder**: Combines acoustic and language models to produce text

#### 2. Voice Activity Detection (VAD)
- Detects presence of human speech in audio stream
- Reduces computational load by processing only speech segments
- Filters out silence and background noise

#### 3. Real-time Processing Challenges
- **Latency**: Minimizing time from speech to recognized text
- **Throughput**: Handling continuous audio streams efficiently
- **Resource management**: Balancing accuracy with computational requirements

## Implementation

### Prerequisites
- ROS 2 Humble with Python support
- OpenAI Whisper installed (`pip install openai-whisper`)
- PyTorch (Whisper dependency)
- Audio input device (microphone)
- CUDA-compatible GPU (optional, for acceleration)

### 1. Whisper Installation and Setup

#### Installing Whisper
```bash
# Create virtual environment
python3 -m venv whisper_env
source whisper_env/bin/activate  # On Windows: whisper_env\Scripts\activate

# Install Whisper
pip install openai-whisper

# For GPU acceleration (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install sounddevice pyaudio numpy scipy
```

#### Basic Whisper Usage
```python
import whisper

# Load different model sizes based on performance requirements:
# tiny (~32MB, fastest) -> base (~74MB) -> small (~244MB) -> medium (~769MB) -> large (~1.5GB)

model = whisper.load_model("small")  # Balanced choice for robotics

# Transcribe audio
result = model.transcribe("audio.wav")
print(result["text"])
```

### 2. ROS 2 Whisper Node Implementation

#### Basic Whisper Node
```python
#!/usr/bin/env python3
"""
ROS 2 node for Whisper-based speech recognition.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
from audio_common_msgs.msg import AudioDataStamped
import whisper
import torch
import numpy as np
import pyaudio
import wave
import threading
import queue
import time
from collections import deque
import tempfile
import os


class WhisperRecognizer(Node):
    """
    ROS 2 node that uses Whisper for speech recognition.
    """

    def __init__(self):
        super().__init__('whisper_recognizer')

        # Parameters
        self.declare_parameter('model_size', 'small')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_size', 1024)
        self.declare_parameter('language', 'en')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('publish_audio', False)
        self.declare_parameter('voice_activity_threshold', 0.02)

        self.model_size = self.get_parameter('model_size').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.language = self.get_parameter('language').value
        self.device = self.get_parameter('device').value
        self.publish_audio = self.get_parameter('publish_audio').value
        self.voice_activity_threshold = self.get_parameter('voice_activity_threshold').value

        # Initialize Whisper model
        self.get_logger().info(f'Loading Whisper model: {self.model_size}')
        self.model = whisper.load_model(self.model_size).to(self.device)
        self.get_logger().info('Whisper model loaded successfully')

        # Publishers
        self.text_pub = self.create_publisher(String, '/speech/text', 10)

        if self.publish_audio:
            self.audio_pub = self.create_publisher(AudioDataStamped, '/speech/audio', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioDataStamped,
            '/audio/input',
            self.audio_callback,
            10
        )

        # Audio processing variables
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 2))  # 2-second buffer
        self.recording_lock = threading.Lock()
        self.is_recording = False
        self.recording_thread = None

        # Start audio processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        self.processing_thread.start()

        self.get_logger().info('Whisper recognizer initialized')

    def audio_callback(self, msg):
        """
        Process incoming audio data from microphone.
        """
        # Convert audio data to numpy array
        audio_data = np.frombuffer(msg.data, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        # Add to buffer
        with self.recording_lock:
            for sample in audio_float:
                self.audio_buffer.append(sample)

    def process_audio_stream(self):
        """
        Continuously process audio stream for speech recognition.
        """
        while rclpy.ok():
            if len(self.audio_buffer) > self.sample_rate * 0.5:  # At least 0.5 seconds of audio
                # Check for voice activity
                audio_segment = list(self.audio_buffer)[-int(self.sample_rate):]  # Last 1 second

                if self.detect_voice_activity(audio_segment):
                    # Extract speech segment (last 3 seconds of audio)
                    speech_segment = list(self.audio_buffer)[-int(self.sample_rate * 3):]

                    if len(speech_segment) > self.sample_rate:  # At least 1 second of speech
                        self.recognize_speech(speech_segment)

                        # Clear buffer to avoid repeated recognition
                        with self.recording_lock:
                            self.audio_buffer.clear()

            time.sleep(0.1)  # 10Hz processing rate

    def detect_voice_activity(self, audio_segment):
        """
        Simple voice activity detection based on energy threshold.
        """
        if len(audio_segment) == 0:
            return False

        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(np.square(audio_segment)))
        return energy > self.voice_activity_threshold

    def recognize_speech(self, audio_segment):
        """
        Recognize speech using Whisper model.
        """
        try:
            # Convert to tensor and ensure proper format
            audio_tensor = torch.from_numpy(np.array(audio_segment)).float()

            # Ensure audio is long enough (minimum 0.5 seconds)
            if len(audio_tensor) < self.sample_rate * 0.5:
                # Pad with zeros if too short
                padding_needed = int(self.sample_rate * 0.5) - len(audio_tensor)
                if padding_needed > 0:
                    padding = torch.zeros(padding_needed)
                    audio_tensor = torch.cat([audio_tensor, padding])

            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_tensor.numpy(),
                language=self.language,
                fp16=(self.device == 'cuda'),  # Use fp16 for GPU inference
                temperature=0.0  # Deterministic output
            )

            # Publish recognized text
            if result and 'text' in result and result['text'].strip():
                text_msg = String()
                text_msg.data = result['text'].strip()

                self.text_pub.publish(text_msg)
                self.get_logger().info(f'Recognized: "{text_msg.data}"')

        except Exception as e:
            self.get_logger().error(f'Error in speech recognition: {e}')

    def record_audio_continuously(self):
        """
        Record audio continuously using PyAudio (alternative implementation).
        """
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        self.get_logger().info('Starting audio recording...')

        try:
            while rclpy.ok() and self.is_recording:
                # Read audio data
                data = stream.read(self.chunk_size)
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Add to processing queue
                self.audio_queue.put(audio_float)

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main(args=None):
    rclpy.init(args=args)

    recognizer = WhisperRecognizer()

    try:
        rclpy.spin(recognizer)
    except KeyboardInterrupt:
        recognizer.get_logger().info('Shutting down Whisper recognizer...')
    finally:
        recognizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 3. Advanced Whisper Configuration

#### Optimized Whisper Node with VAD and Streaming
```python
#!/usr/bin/env python3
"""
Advanced Whisper node with streaming recognition and voice activity detection.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Twist
import whisper
import torch
import numpy as np
import threading
import queue
import time
from collections import deque
import webrtcvad  # Voice Activity Detection
import struct
import collections


class AdvancedWhisperRecognizer(Node):
    """
    Advanced Whisper node with streaming recognition and VAD.
    """

    def __init__(self):
        super().__init__('advanced_whisper_recognizer')

        # Parameters
        self.declare_parameter('model_size', 'medium')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('vad_aggressiveness', 2)  # 0-3, higher = more aggressive
        self.declare_parameter('silence_threshold', 0.5)  # seconds of silence to trigger recognition
        self.declare_parameter('min_speech_duration', 0.5)  # minimum speech duration
        self.declare_parameter('max_buffer_duration', 10.0)  # maximum buffer duration

        self.model_size = self.get_parameter('model_size').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.vad_aggressiveness = self.get_parameter('vad_aggressiveness').value
        self.silence_threshold = self.get_parameter('silence_threshold').value
        self.min_speech_duration = self.get_parameter('min_speech_duration').value
        self.max_buffer_duration = self.get_parameter('max_buffer_duration').value

        # Initialize Whisper model
        self.get_logger().info(f'Loading Whisper model: {self.model_size}')
        self.model = whisper.load_model(self.model_size).to(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.get_logger().info('Whisper model loaded successfully')

        # Initialize VAD
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)

        # Publishers
        self.text_pub = self.create_publisher(String, '/speech/text', 10)
        self.command_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.recording_status_pub = self.create_publisher(Bool, '/speech/recording_status', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio/input',
            self.audio_callback,
            10
        )

        # State variables
        self.audio_buffer = collections.deque(maxlen=int(self.sample_rate * self.max_buffer_duration))
        self.speech_segments = []
        self.is_speaking = False
        self.silence_start_time = None
        self.speech_start_time = None

        # Recognition lock
        self.recognition_lock = threading.Lock()

        # Start recognition thread
        self.recognition_thread = threading.Thread(target=self.recognition_worker, daemon=True)
        self.recognition_thread.start()

        self.get_logger().info('Advanced Whisper recognizer initialized')

    def audio_callback(self, msg):
        """
        Process incoming audio chunks with VAD.
        """
        # Convert audio data to 16-bit PCM
        audio_data = msg.data
        chunk_duration = len(audio_data) / (2 * self.sample_rate)  # 2 bytes per sample

        # Process in 10ms chunks for VAD (required by WebRTC VAD)
        chunk_size = int(0.01 * self.sample_rate * 2)  # 10ms in bytes

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if len(chunk) == chunk_size:  # Full chunk
                # Check for voice activity
                is_speech = self.vad.is_speech(chunk, self.sample_rate)

                if is_speech:
                    # Add to buffer
                    chunk_audio = struct.unpack('<' + 'h'*(len(chunk)//2), chunk)
                    self.audio_buffer.extend(chunk_audio)

                    if not self.is_speaking:
                        # Start of speech
                        self.is_speaking = True
                        self.speech_start_time = time.time()
                        self.silence_start_time = None

                        # Publish recording status
                        status_msg = Bool()
                        status_msg.data = True
                        self.recording_status_pub.publish(status_msg)

                        self.get_logger().info('Speech detected, starting recording...')
                else:
                    # Silence detected
                    if self.is_speaking:
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                        else:
                            # Check if silence duration exceeds threshold
                            silence_duration = time.time() - self.silence_start_time
                            speech_duration = time.time() - self.speech_start_time

                            if (silence_duration >= self.silence_threshold and
                                speech_duration >= self.min_speech_duration):
                                # Prepare speech segment for recognition
                                self.prepare_speech_segment()

                                # Reset state
                                self.is_speaking = False
                                self.silence_start_time = None
                                self.speech_start_time = None

                                # Publish recording status
                                status_msg = Bool()
                                status_msg.data = False
                                self.recording_status_pub.publish(status_msg)

                                self.get_logger().info('Silence detected, preparing for recognition...')
                    else:
                        # Add silence to buffer to prevent overflow
                        chunk_audio = struct.unpack('<' + 'h'*(len(chunk)//2), chunk)
                        self.audio_buffer.extend(chunk_audio)

    def prepare_speech_segment(self):
        """
        Prepare speech segment for recognition.
        """
        if len(self.audio_buffer) == 0:
            return

        # Convert buffer to numpy array
        audio_array = np.array(list(self.audio_buffer), dtype=np.float32) / 32768.0  # Normalize

        # Only process if we have enough audio
        if len(audio_array) > self.sample_rate * self.min_speech_duration:
            # Queue for recognition
            with self.recognition_lock:
                self.speech_segments.append(audio_array.copy())

            # Clear buffer for next segment
            self.audio_buffer.clear()

    def recognition_worker(self):
        """
        Worker thread for speech recognition.
        """
        while rclpy.ok():
            if len(self.speech_segments) > 0:
                with self.recognition_lock:
                    if len(self.speech_segments) > 0:
                        speech_segment = self.speech_segments.pop(0)

                self.perform_recognition(speech_segment)
            else:
                time.sleep(0.1)  # Sleep if no segments to process

    def perform_recognition(self, audio_segment):
        """
        Perform speech recognition on audio segment.
        """
        try:
            # Ensure minimum length
            min_samples = int(self.sample_rate * 0.5)  # 0.5 seconds minimum
            if len(audio_segment) < min_samples:
                # Pad with zeros
                padding_needed = min_samples - len(audio_segment)
                padding = np.zeros(padding_needed, dtype=np.float32)
                audio_segment = np.concatenate([audio_segment, padding])

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_segment).to(
                self.model.device if hasattr(self.model, 'device') else 'cpu'
            )

            # Transcribe with specific options for command recognition
            result = self.model.transcribe(
                audio_tensor,
                language=self.get_parameter('language').value if self.has_parameter('language') else 'en',
                task='transcribe',
                fp16=torch.cuda.is_available(),
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.0,  # Filter out low-quality transcriptions
                logprob_threshold=-1.0,  # Filter out low-confidence transcriptions
                no_speech_threshold=0.6  # Filter out non-speech
            )

            # Process result
            if result and 'text' in result:
                recognized_text = result['text'].strip()

                if recognized_text:  # Only publish if we got actual text
                    # Publish recognized text
                    text_msg = String()
                    text_msg.data = recognized_text
                    self.text_pub.publish(text_msg)

                    self.get_logger().info(f'Recognized: "{recognized_text}"')

                    # Process command if it's a robot command
                    self.process_robot_command(recognized_text)
                else:
                    self.get_logger().info('Recognition returned empty text')
            else:
                self.get_logger().info('No recognition result obtained')

        except Exception as e:
            self.get_logger().error(f'Error in speech recognition: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def process_robot_command(self, text):
        """
        Process recognized text as robot commands.
        """
        # Convert to lowercase for easier matching
        command = text.lower().strip()

        # Define command mappings
        command_mappings = {
            'move forward': self.move_forward,
            'move backward': self.move_backward,
            'turn left': self.turn_left,
            'turn right': self.turn_right,
            'stop': self.stop_robot,
            'go forward': self.move_forward,
            'go back': self.move_backward,
            'rotate left': self.turn_left,
            'rotate right': self.turn_right,
            'halt': self.stop_robot,
        }

        # Find matching command
        for cmd_text, cmd_func in command_mappings.items():
            if cmd_text in command:
                cmd_func()
                self.get_logger().info(f'Executed command: {cmd_text}')
                return

        # If no specific command found, log as general text
        self.get_logger().info(f'Unrecognized command: {command}')

    def move_forward(self):
        """Move robot forward."""
        cmd = Twist()
        cmd.linear.x = 0.2  # m/s
        cmd.angular.z = 0.0
        self.command_pub.publish(cmd)

    def move_backward(self):
        """Move robot backward."""
        cmd = Twist()
        cmd.linear.x = -0.2  # m/s
        cmd.angular.z = 0.0
        self.command_pub.publish(cmd)

    def turn_left(self):
        """Turn robot left."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.3  # rad/s
        self.command_pub.publish(cmd)

    def turn_right(self):
        """Turn robot right."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.3  # rad/s
        self.command_pub.publish(cmd)

    def stop_robot(self):
        """Stop robot movement."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.command_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    recognizer = AdvancedWhisperRecognizer()

    try:
        rclpy.spin(recognizer)
    except KeyboardInterrupt:
        recognizer.get_logger().info('Shutting down advanced Whisper recognizer...')
    finally:
        recognizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 4. Whisper Model Optimization

#### Quantized Whisper for Edge Deployment
```python
#!/usr/bin/env python3
"""
Optimized Whisper node for edge deployment with quantization.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import torch
import numpy as np
from collections import deque
import time


class OptimizedWhisperNode(Node):
    """
    Optimized Whisper node for edge deployment with quantization.
    """

    def __init__(self):
        super().__init__('optimized_whisper')

        # Parameters for optimization
        self.declare_parameter('model_size', 'tiny')
        self.declare_parameter('quantize_model', True)
        self.declare_parameter('use_int8', True)
        self.declare_parameter('batch_size', 1)
        self.declare_parameter('sample_rate', 16000)

        self.model_size = self.get_parameter('model_size').value
        self.quantize_model = self.get_parameter('quantize_model').value
        self.use_int8 = self.get_parameter('use_int8').value
        self.batch_size = self.get_parameter('batch_size').value
        self.sample_rate = self.get_parameter('sample_rate').value

        # Load optimized model
        self.load_optimized_model()

        # Publishers and subscribers
        self.text_pub = self.create_publisher(String, '/speech/text', 10)

        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 5))  # 5-second buffer

        self.get_logger().info('Optimized Whisper node initialized')

    def load_optimized_model(self):
        """
        Load Whisper model with optimizations for edge deployment.
        """
        self.get_logger().info(f'Loading optimized Whisper model: {self.model_size}')

        # Load model
        self.model = whisper.load_model(self.model_size)

        if self.quantize_model:
            # Apply dynamic quantization to reduce model size and improve inference speed
            if self.use_int8:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear, torch.nn.Conv1d},
                    dtype=torch.qint8
                )
                self.get_logger().info('Applied INT8 quantization')
            else:
                # Use float16 for GPU
                if torch.cuda.is_available():
                    self.model = self.model.half()
                    self.get_logger().info('Applied FP16 quantization')

        # Move to appropriate device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)

        self.get_logger().info('Optimized Whisper model loaded successfully')

    def transcribe_with_performance_optimization(self, audio_tensor):
        """
        Transcribe audio with performance optimizations.
        """
        # Apply optimizations during inference
        with torch.no_grad():  # Disable gradient computation
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Synchronize GPU operations

            start_time = time.time()

            # Perform transcription with optimized settings
            result = self.model.transcribe(
                audio_tensor,
                fp16=(torch.cuda.is_available() and not self.use_int8),
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                best_of=1,  # Reduce search complexity
                beam_size=1,  # Greedy decoding for speed
            )

            inference_time = time.time() - start_time

            self.get_logger().info(f'Inference time: {inference_time:.3f}s')

            return result

    def recognize_speech_optimized(self, audio_segment):
        """
        Recognize speech with performance optimizations.
        """
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_segment).float()

            # Ensure minimum length
            min_samples = int(self.sample_rate * 0.5)
            if len(audio_tensor) < min_samples:
                padding_needed = min_samples - len(audio_tensor)
                padding = torch.zeros(padding_needed)
                audio_tensor = torch.cat([audio_tensor, padding])

            # Perform optimized recognition
            result = self.transcribe_with_performance_optimization(audio_tensor.numpy())

            if result and 'text' in result:
                recognized_text = result['text'].strip()

                if recognized_text:
                    text_msg = String()
                    text_msg.data = recognized_text
                    self.text_pub.publish(text_msg)

                    self.get_logger().info(f'Recognized: "{recognized_text}"')

                    return recognized_text

        except Exception as e:
            self.get_logger().error(f'Error in optimized recognition: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)

    node = OptimizedWhisperNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down optimized Whisper node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 5. Whisper Integration with ROS 2 Launch Files

#### Launch File for Whisper Node
```xml
<!-- launch/whisper_recognition.launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    model_size_arg = DeclareLaunchArgument(
        'model_size',
        default_value='small',
        description='Whisper model size (tiny, base, small, medium, large)'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda' if torch.cuda.is_available() else 'cpu',
        description='Device to run Whisper on (cpu or cuda)'
    )

    sample_rate_arg = DeclareLaunchArgument(
        'sample_rate',
        default_value='16000',
        description='Audio sample rate'
    )

    # Whisper recognizer node
    whisper_node = Node(
        package='robot_speech',
        executable='whisper_recognizer',
        name='whisper_recognizer',
        parameters=[
            {
                'model_size': LaunchConfiguration('model_size'),
                'device': LaunchConfiguration('device'),
                'sample_rate': LaunchConfiguration('sample_rate'),
                'language': 'en',
                'voice_activity_threshold': 0.02
            }
        ],
        remappings=[
            ('/audio/input', '/microphone/audio_raw'),
            ('/speech/text', '/voice_commands'),
        ]
    )

    # Optional: Audio input node (if needed)
    audio_input_node = Node(
        package='audio_capture',
        executable='audio_capture_node',
        name='audio_input',
        parameters=[
            {
                'device': -1,  # Default device
                'sample_rate': LaunchConfiguration('sample_rate'),
                'channels': 1,
                'chunk_size': 1024
            }
        ]
    )

    return LaunchDescription([
        model_size_arg,
        device_arg,
        sample_rate_arg,
        whisper_node,
        audio_input_node
    ])
```

### 6. Performance Optimization Techniques

#### Whisper Performance Profiler
```python
#!/usr/bin/env python3
"""
Performance profiler for Whisper-based speech recognition.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import whisper
import torch
import numpy as np
import time
import psutil
from std_msgs.msg import Float32
import threading


class WhisperPerformanceProfiler(Node):
    """
    Profile Whisper performance metrics for robotics applications.
    """

    def __init__(self):
        super().__init__('whisper_performance_profiler')

        # Parameters
        self.declare_parameter('model_size', 'small')
        self.declare_parameter('profile_interval', 10.0)  # seconds

        self.model_size = self.get_parameter('model_size').value
        self.profile_interval = self.get_parameter('profile_interval').value

        # Initialize model
        self.model = whisper.load_model(self.model_size).to(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Publishers for metrics
        self.latency_pub = self.create_publisher(Float32, '/speech/latency', 10)
        self.cpu_usage_pub = self.create_publisher(Float32, '/speech/cpu_usage', 10)
        self.memory_usage_pub = self.create_publisher(Float32, '/speech/memory_usage', 10)
        self.power_usage_pub = self.create_publisher(Float32, '/speech/power_usage', 10)

        # Profiling variables
        self.inference_times = []
        self.profile_timer = self.create_timer(self.profile_interval, self.publish_profile_metrics)

        self.get_logger().info('Whisper performance profiler initialized')

    def profile_inference(self, audio_tensor):
        """
        Profile inference performance.
        """
        start_time = time.time()

        # Monitor system resources
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.virtual_memory().percent

        # Perform inference
        result = self.model.transcribe(audio_tensor)

        inference_time = time.time() - start_time

        # Monitor system resources after
        cpu_after = psutil.cpu_percent()
        memory_after = psutil.virtual_memory().percent

        # Store metrics
        self.inference_times.append(inference_time)

        # Publish metrics
        latency_msg = Float32()
        latency_msg.data = inference_time
        self.latency_pub.publish(latency_msg)

        cpu_msg = Float32()
        cpu_msg.data = max(cpu_before, cpu_after)
        self.cpu_usage_pub.publish(cpu_msg)

        memory_msg = Float32()
        memory_msg.data = max(memory_before, memory_after)
        self.memory_usage_pub.publish(memory_msg)

        return result

    def publish_profile_metrics(self):
        """
        Publish aggregated performance metrics.
        """
        if self.inference_times:
            avg_latency = np.mean(self.inference_times)
            min_latency = np.min(self.inference_times)
            max_latency = np.max(self.inference_times)

            self.get_logger().info(
                f'Performance Metrics (last {len(self.inference_times)} inferences):\n'
                f'  Avg Latency: {avg_latency:.3f}s\n'
                f'  Min Latency: {min_latency:.3f}s\n'
                f'  Max Latency: {max_latency:.3f}s\n'
                f'  Throughput: {1.0/avg_latency:.2f} inferences/sec'
            )

            # Clear for next interval
            self.inference_times.clear()


def main(args=None):
    rclpy.init(args=args)

    profiler = WhisperPerformanceProfiler()

    try:
        rclpy.spin(profiler)
    except KeyboardInterrupt:
        profiler.get_logger().info('Shutting down performance profiler...')
    finally:
        profiler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercises

1. Implement the basic Whisper recognizer node and test with audio files
2. Integrate VAD (Voice Activity Detection) to improve real-time performance
3. Optimize the Whisper model for deployment on embedded systems
4. Create a launch file that starts the Whisper node with proper parameters
5. Implement command recognition for robot navigation commands
6. Profile the performance of different Whisper model sizes
7. Test the system with various audio input devices and noise conditions
8. Create a fallback mechanism when Whisper recognition fails

## References

1. OpenAI Whisper GitHub: https://github.com/openai/whisper
2. Whisper Paper: https://cdn.openai.com/papers/whisper.pdf
3. WebRTC VAD: https://github.com/wiseman/py-webrtcvad
4. ROS 2 Audio Common: http://wiki.ros.org/audio_common

## Further Reading

- Advanced audio preprocessing for robotics applications
- Integration with other ASR systems (DeepSpeech, Vosk)
- Multi-language support for international robotics applications
- Privacy considerations for voice processing in robotics
- Edge AI optimization techniques for speech recognition
- Integration with natural language understanding systems