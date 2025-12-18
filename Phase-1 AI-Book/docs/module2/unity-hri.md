---
sidebar_position: 24
---

# Unity Integration Pipeline for Human-Robot Interaction

## Learning Objectives
By the end of this module, students will be able to:
- Set up Unity 3D for robotics simulation and human-robot interaction
- Create realistic apartment environments with interactive objects
- Implement data pipelines for sensor data visualization
- Synchronize robot states between ROS 2 and Unity
- Develop human-robot interaction scenarios in Unity
- Integrate Unity with Gazebo for hybrid simulation

## Theory

### Unity in Robotics Applications
Unity is a powerful 3D development platform that can be used for robotics simulation, visualization, and human-robot interaction prototyping. It provides:
- High-quality 3D graphics rendering
- Physics simulation capabilities
- User interface development tools
- Cross-platform deployment options
- Asset ecosystem for 3D models and environments

### Key Concepts
- **Robotics Middleware Interface**: Connecting Unity to ROS 2 or other robotics frameworks
- **Asset Pipeline**: Managing 3D models, materials, and textures for robot and environment
- **Physics Integration**: Ensuring Unity physics aligns with real-world or Gazebo physics
- **Real-time Visualization**: Displaying sensor data and robot states in Unity
- **Interaction Design**: Creating intuitive interfaces for human-robot interaction

### Unity-Rosbridge Integration
The most common approach for connecting Unity to ROS is through rosbridge, which allows JSON-based communication over WebSocket connections.

## Implementation

### Prerequisites
- Unity 2022 LTS or higher
- Unity Rosbridge Client package
- ROS 2 Humble with rosbridge
- Basic knowledge of Unity development and C#

### Setting Up Unity for Robotics

#### 1. Installing Unity Rosbridge Client
First, install the Unity Rosbridge Client package:

1. Open Unity Hub and create a new 3D project
2. In the Package Manager (Window > Package Manager):
   - Click the "+" button in the top-left corner
   - Select "Add package from git URL..."
   - Enter the Unity Rosbridge Client repository URL
   - Install the package

Alternatively, you can install via the manifest file (`Packages/manifest.json`):

```json
{
  "dependencies": {
    "com.unity.robotics.ros-bridge": "https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-bridge#v0.6.0",
    "com.unity.robotics.urdf-importer": "https://github.com/Unity-Technologies/Unity-Robotics-Hub.git?path=/submodules/URDF-Importer#v0.6.0"
  }
}
```

#### 2. Basic Unity Scene Setup
Create a basic scene structure for robotics applications:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;

public class RobotVisualization : MonoBehaviour
{
    [SerializeField]
    private string rosIP = "127.0.0.1";
    [SerializeField]
    private int rosPort = 9090;

    private ROSConnection ros;
    private string robotTopic = "/robot_state";

    void Start()
    {
        // Connect to ROS
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);

        // Subscribe to robot state topic
        ros.Subscribe<JointStateMsg>(robotTopic, UpdateRobotState);
    }

    void UpdateRobotState(JointStateMsg jointState)
    {
        // Process joint state message
        for (int i = 0; i < jointState.name_array.Length; i++)
        {
            string jointName = jointState.name_array[i];
            float jointPosition = jointState.position_array[i];

            // Update corresponding joint in Unity
            Transform jointTransform = transform.Find(jointName);
            if (jointTransform != null)
            {
                // Example: rotate a revolute joint
                jointTransform.Rotate(Vector3.up, jointPosition * Mathf.Rad2Deg);
            }
        }
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros.Close();
        }
    }
}
```

### Creating Apartment Environment in Unity

#### 1. Basic Apartment Layout
Create a basic apartment environment with multiple rooms:

```csharp
using UnityEngine;

public class ApartmentBuilder : MonoBehaviour
{
    [Header("Room Dimensions")]
    public Vector3 livingRoomSize = new Vector3(5f, 4f, 3f);
    public Vector3 kitchenSize = new Vector3(3f, 3f, 3f);
    public Vector3 bedroomSize = new Vector3(4f, 4f, 3f);

    [Header("Furniture Prefabs")]
    public GameObject[] furniturePrefabs;

    void Start()
    {
        BuildApartment();
    }

    void BuildApartment()
    {
        // Create main room structure
        CreateRoom("LivingRoom", Vector3.zero, livingRoomSize);

        // Create kitchen attached to living room
        Vector3 kitchenPosition = new Vector3(livingRoomSize.x / 2 + kitchenSize.x / 2, 0, 0);
        CreateRoom("Kitchen", kitchenPosition, kitchenSize);

        // Create bedroom attached to living room
        Vector3 bedroomPosition = new Vector3(0, 0, livingRoomSize.z / 2 + bedroomSize.z / 2);
        CreateRoom("Bedroom", bedroomPosition, bedroomSize);

        // Add furniture
        PlaceFurniture();
    }

    GameObject CreateRoom(string roomName, Vector3 position, Vector3 size)
    {
        GameObject room = new GameObject(roomName);
        room.transform.position = position;

        // Create floor
        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
        floor.transform.parent = room.transform;
        floor.transform.localScale = new Vector3(size.x, 0.1f, size.z);
        floor.transform.localPosition = Vector3.zero;
        floor.GetComponent<Renderer>().material.color = Color.gray;

        // Create walls
        CreateWall(room.transform, new Vector3(0, size.y/2, size.z/2), new Vector3(size.x, size.y, 0.1f)); // Back wall
        CreateWall(room.transform, new Vector3(0, size.y/2, -size.z/2), new Vector3(size.x, size.y, 0.1f)); // Front wall
        CreateWall(room.transform, new Vector3(size.x/2, size.y/2, 0), new Vector3(0.1f, size.y, size.z)); // Right wall
        CreateWall(room.transform, new Vector3(-size.x/2, size.y/2, 0), new Vector3(0.1f, size.y, size.z)); // Left wall

        return room;
    }

    GameObject CreateWall(Transform parent, Vector3 localPosition, Vector3 scale)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.transform.parent = parent;
        wall.transform.localPosition = localPosition;
        wall.transform.localScale = scale;
        wall.GetComponent<Renderer>().material.color = Color.white;
        return wall;
    }

    void PlaceFurniture()
    {
        // Place furniture based on room layout
        foreach (GameObject prefab in furniturePrefabs)
        {
            if (prefab != null)
            {
                GameObject furniture = Instantiate(prefab);

                // Random placement within bounds
                Vector3 randomPos = new Vector3(
                    Random.Range(-2f, 2f),
                    0,
                    Random.Range(-1.5f, 1.5f)
                );

                furniture.transform.position = randomPos;
            }
        }
    }
}
```

#### 2. Advanced Apartment Environment with Materials
```csharp
using UnityEngine;

[CreateAssetMenu(fileName = "ApartmentConfig", menuName = "Robotics/Apartment Configuration")]
public class ApartmentConfiguration : ScriptableObject
{
    [Header("Material Settings")]
    public Material floorMaterial;
    public Material wallMaterial;
    public Material furnitureMaterial;

    [Header("Lighting")]
    public Color ambientColor = new Color(0.4f, 0.4f, 0.4f, 1f);
    public float intensityMultiplier = 1.0f;

    [Header("Physics")]
    public PhysicMaterial floorPhysicsMaterial;
    public float gravityScale = 1.0f;
}

public class AdvancedApartmentBuilder : MonoBehaviour
{
    [SerializeField] private ApartmentConfiguration config;
    [SerializeField] private GameObject[] furniturePrefabs;
    [SerializeField] private Light[] roomLights;

    void Start()
    {
        ApplyConfiguration();
        BuildDetailedEnvironment();
    }

    void ApplyConfiguration()
    {
        RenderSettings.ambientLight = config.ambientColor;
        RenderSettings.ambientIntensity = config.intensityMultiplier;

        if (roomLights != null)
        {
            foreach (Light light in roomLights)
            {
                light.intensity *= config.intensityMultiplier;
            }
        }
    }

    void BuildDetailedEnvironment()
    {
        // Create detailed apartment with proper materials
        CreateDetailedRooms();
        SetupPhysics();
        AddInteractiveElements();
    }

    void CreateDetailedRooms()
    {
        // Create rooms with proper materials
        GameObject livingRoom = CreateRoomWithMaterials("LivingRoom", Vector3.zero, new Vector3(5f, 3f, 4f));
        GameObject kitchen = CreateRoomWithMaterials("Kitchen", new Vector3(3f, 0, 0), new Vector3(3f, 3f, 3f));
        GameObject bedroom = CreateRoomWithMaterials("Bedroom", new Vector3(0, 0, 3f), new Vector3(4f, 3f, 3f));

        // Add furniture with proper materials
        PlaceFurnitureWithMaterials(livingRoom.transform, furniturePrefabs);
    }

    GameObject CreateRoomWithMaterials(string name, Vector3 position, Vector3 size)
    {
        GameObject room = new GameObject(name);
        room.transform.position = position;

        // Create floor with material
        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.transform.parent = room.transform;
        floor.transform.localScale = new Vector3(size.x / 10f, 1, size.z / 10f); // Plane is 10x10 units
        floor.transform.localPosition = Vector3.zero;
        floor.GetComponent<Renderer>().material = config.floorMaterial;

        // Create walls with material
        CreateTexturedWall(room.transform, Vector3.forward * size.z/2, new Vector3(size.x, size.y, 0.2f), config.wallMaterial);
        CreateTexturedWall(room.transform, Vector3.back * size.z/2, new Vector3(size.x, size.y, 0.2f), config.wallMaterial);
        CreateTexturedWall(room.transform, Vector3.right * size.x/2, new Vector3(0.2f, size.y, size.z), config.wallMaterial);
        CreateTexturedWall(room.transform, Vector3.left * size.x/2, new Vector3(0.2f, size.y, size.z), config.wallMaterial);

        return room;
    }

    GameObject CreateTexturedWall(Transform parent, Vector3 localPosition, Vector3 scale, Material material)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.transform.parent = parent;
        wall.transform.localPosition = localPosition;
        wall.transform.localScale = scale;
        wall.GetComponent<Renderer>().material = material;
        wall.GetComponent<BoxCollider>().enabled = true;
        return wall;
    }

    void PlaceFurnitureWithMaterials(Transform roomParent, GameObject[] furniture)
    {
        foreach (GameObject furnitureItem in furniture)
        {
            if (furnitureItem != null)
            {
                GameObject placedFurniture = Instantiate(furnitureItem, roomParent);

                // Apply material to furniture
                Renderer[] renderers = placedFurniture.GetComponentsInChildren<Renderer>();
                foreach (Renderer renderer in renderers)
                {
                    renderer.material = config.furnitureMaterial;
                }

                // Position randomly within room bounds
                placedFurniture.transform.localPosition = new Vector3(
                    Random.Range(-2f, 2f),
                    0,
                    Random.Range(-1.5f, 1.5f)
                );
            }
        }
    }

    void SetupPhysics()
    {
        // Configure physics for the environment
        Physics.gravity = new Vector3(0, -9.81f * config.gravityScale, 0);
    }

    void AddInteractiveElements()
    {
        // Add interactive objects for human-robot interaction
        GameObject[] interactiveObjects = GameObject.FindGameObjectsWithTag("Interactive");
        foreach (GameObject obj in interactiveObjects)
        {
            // Add interaction scripts
            if (!obj.GetComponent<InteractiveObject>())
            {
                obj.AddComponent<InteractiveObject>();
            }
        }
    }
}
```

### Sensor Data Visualization in Unity

#### 1. LiDAR Point Cloud Visualization
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class LidarVisualizer : MonoBehaviour
{
    [SerializeField] private LineRenderer pointCloudRenderer;
    [SerializeField] private int maxPoints = 1080;
    [SerializeField] private float pointSize = 0.02f;
    [SerializeField] private Color pointColor = Color.red;

    private List<GameObject> pointCloudObjects = new List<GameObject>();

    void Start()
    {
        if (pointCloudRenderer == null)
        {
            pointCloudRenderer = GetComponent<LineRenderer>();
        }

        // Initialize point cloud renderer
        if (pointCloudRenderer != null)
        {
            pointCloudRenderer.positionCount = maxPoints;
            pointCloudRenderer.startWidth = pointSize;
            pointCloudRenderer.endWidth = pointSize;
            pointCloudRenderer.startColor = pointColor;
            pointCloudRenderer.endColor = pointColor;
        }

        // Subscribe to LiDAR data
        ROSConnection.GetOrCreateInstance().Subscribe<LaserScanMsg>(
            "/scan",
            ProcessLidarData
        );
    }

    void ProcessLidarData(LaserScanMsg scan)
    {
        if (scan.ranges.Length == 0) return;

        Vector3[] points = new Vector3[scan.ranges.Length];
        float angle = scan.angle_min;

        for (int i = 0; i < scan.ranges.Length; i++)
        {
            float range = scan.ranges[i];

            if (range >= scan.range_min && range <= scan.range_max)
            {
                // Calculate point position in 3D space
                float x = range * Mathf.Cos(angle);
                float y = 0f; // Height is 0 for 2D LiDAR
                float z = range * Mathf.Sin(angle);

                points[i] = new Vector3(x, y, z);
            }
            else
            {
                // Invalid range, set to zero or use infinity
                points[i] = Vector3.zero;
            }

            angle += scan.angle_increment;
        }

        // Update visualization
        UpdatePointCloudVisualization(points);
    }

    void UpdatePointCloudVisualization(Vector3[] points)
    {
        if (pointCloudRenderer != null)
        {
            // Limit to max points if necessary
            int pointCount = Mathf.Min(points.Length, maxPoints);
            pointCloudRenderer.positionCount = pointCount;

            // Set positions
            for (int i = 0; i < pointCount; i++)
            {
                pointCloudRenderer.SetPosition(i, points[i]);
            }
        }

        // Alternative: Create individual point objects for more control
        UpdateIndividualPoints(points);
    }

    void UpdateIndividualPoints(Vector3[] points)
    {
        // Destroy old points
        foreach (GameObject pointObj in pointCloudObjects)
        {
            if (pointObj != null)
            {
                DestroyImmediate(pointObj);
            }
        }
        pointCloudObjects.Clear();

        // Create new points
        foreach (Vector3 point in points)
        {
            if (point != Vector3.zero) // Skip invalid points
            {
                GameObject pointObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                pointObj.transform.position = transform.TransformPoint(point);
                pointObj.transform.localScale = Vector3.one * pointSize;
                pointObj.GetComponent<Renderer>().material.color = pointColor;

                // Make it a child of this object
                pointObj.transform.SetParent(transform);

                pointCloudObjects.Add(pointObj);
            }
        }
    }
}
```

#### 2. RGB-D Camera Visualization
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using System.IO;

public class RgbdCameraVisualizer : MonoBehaviour
{
    [SerializeField] private Camera unityCamera;
    [SerializeField] private Renderer cameraDisplayRenderer;
    [SerializeField] private Shader depthShader;

    private Texture2D rgbTexture;
    private Texture2D depthTexture;
    private byte[] imageDataBuffer;

    void Start()
    {
        // Initialize camera display
        if (cameraDisplayRenderer != null && unityCamera != null)
        {
            RenderTexture renderTexture = new RenderTexture(640, 480, 24);
            unityCamera.targetTexture = renderTexture;
            cameraDisplayRenderer.material.mainTexture = renderTexture;
        }

        // Subscribe to RGB and Depth topics
        ROSConnection.GetOrCreateInstance().Subscribe<ImageMsg>(
            "/camera/rgb/image_raw",
            ProcessRgbImage
        );

        ROSConnection.GetOrCreateInstance().Subscribe<ImageMsg>(
            "/camera/depth/image_raw",
            ProcessDepthImage
        );
    }

    void ProcessRgbImage(ImageMsg imageMsg)
    {
        if (imageMsg.encoding.Equals("rgb8") || imageMsg.encoding.Equals("bgr8"))
        {
            // Create texture from ROS image data
            int width = (int)imageMsg.width;
            int height = (int)imageMsg.height;

            if (rgbTexture == null || rgbTexture.width != width || rgbTexture.height != height)
            {
                rgbTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
            }

            // Convert ROS image data to Unity texture format
            Color32[] colors = new Color32[width * height];

            if (imageMsg.encoding.Equals("rgb8"))
            {
                for (int i = 0; i < imageMsg.data.Count; i += 3)
                {
                    int pixelIndex = i / 3;
                    if (pixelIndex < colors.Length)
                    {
                        colors[pixelIndex] = new Color32(
                            imageMsg.data[i],     // R
                            imageMsg.data[i + 1], // G
                            imageMsg.data[i + 2], // B
                            255                 // A
                        );
                    }
                }
            }
            else if (imageMsg.encoding.Equals("bgr8"))
            {
                for (int i = 0; i < imageMsg.data.Count; i += 3)
                {
                    int pixelIndex = i / 3;
                    if (pixelIndex < colors.Length)
                    {
                        colors[pixelIndex] = new Color32(
                            imageMsg.data[i + 2], // R (BGR -> RGB)
                            imageMsg.data[i + 1], // G
                            imageMsg.data[i],     // B (BGR -> RGB)
                            255                 // A
                        );
                    }
                }
            }

            rgbTexture.SetPixels32(colors);
            rgbTexture.Apply();

            // Update display if available
            if (cameraDisplayRenderer != null)
            {
                cameraDisplayRenderer.material.mainTexture = rgbTexture;
            }
        }
    }

    void ProcessDepthImage(ImageMsg depthMsg)
    {
        if (depthMsg.encoding.Equals("16UC1") || depthMsg.encoding.Equals("32FC1"))
        {
            int width = (int)depthMsg.width;
            int height = (int)depthMsg.height;

            if (depthTexture == null || depthTexture.width != width || depthTexture.height != height)
            {
                depthTexture = new Texture2D(width, height, TextureFormat.RFloat, false);
            }

            // Process depth data based on encoding
            if (depthMsg.encoding.Equals("16UC1"))
            {
                // 16-bit unsigned integer depth
                Color[] depthColors = new Color[width * height];

                for (int i = 0; i < depthMsg.data.Count; i += 2)
                {
                    if (i + 1 < depthMsg.data.Count)
                    {
                        ushort depthValue = (ushort)(depthMsg.data[i] | (depthMsg.data[i + 1] << 8));
                        float normalizedDepth = (float)depthValue / 65535f; // Normalize to 0-1

                        int pixelIndex = i / 2;
                        if (pixelIndex < depthColors.Length)
                        {
                            depthColors[pixelIndex] = new Color(normalizedDepth, normalizedDepth, normalizedDepth, 1);
                        }
                    }
                }

                depthTexture.SetPixels(depthColors);
            }
            else if (depthMsg.encoding.Equals("32FC1"))
            {
                // 32-bit float depth
                Color[] depthColors = new Color[width * height];

                for (int i = 0; i < depthMsg.data.Count; i += 4)
                {
                    if (i + 3 < depthMsg.data.Count)
                    {
                        byte[] floatBytes = new byte[4];
                        for (int j = 0; j < 4; j++)
                        {
                            floatBytes[j] = depthMsg.data[i + j];
                        }

                        float depthValue = System.BitConverter.ToSingle(floatBytes, 0);
                        float normalizedDepth = Mathf.Clamp01(depthValue / 10.0f); // Normalize to 0-10m range

                        int pixelIndex = i / 4;
                        if (pixelIndex < depthColors.Length)
                        {
                            depthColors[pixelIndex] = new Color(normalizedDepth, normalizedDepth, normalizedDepth, 1);
                        }
                    }
                }

                depthTexture.SetPixels(depthColors);
            }

            depthTexture.Apply();
        }
    }
}
```

### Robot State Synchronization

#### 1. Joint State Synchronization
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class RobotStateSynchronizer : MonoBehaviour
{
    [System.Serializable]
    public class JointMapping
    {
        public string rosJointName;
        public Transform unityJointTransform;
        public JointType jointType = JointType.Revolute;
        public float minAngle = -180f;
        public float maxAngle = 180f;
        public float multiplier = 1f;
    }

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed
    }

    [SerializeField] private List<JointMapping> jointMappings = new List<JointMapping>();
    [SerializeField] private string jointStateTopic = "/joint_states";

    private Dictionary<string, float> jointPositions = new Dictionary<string, float>();

    void Start()
    {
        // Subscribe to joint states
        ROSConnection.GetOrCreateInstance().Subscribe<JointStateMsg>(
            jointStateTopic,
            ProcessJointStates
        );
    }

    void ProcessJointStates(JointStateMsg jointState)
    {
        // Update internal joint position dictionary
        for (int i = 0; i < jointState.name_array.Length; i++)
        {
            if (i < jointState.position_array.Length)
            {
                jointPositions[jointState.name_array[i]] = jointState.position_array[i];
            }
        }

        // Update Unity transforms
        UpdateUnityJoints();
    }

    void UpdateUnityJoints()
    {
        foreach (JointMapping mapping in jointMappings)
        {
            if (jointPositions.ContainsKey(mapping.rosJointName))
            {
                float rosPosition = jointPositions[mapping.rosJointName];
                float unityPosition = rosPosition * mapping.multiplier;

                switch (mapping.jointType)
                {
                    case JointType.Revolute:
                        // Rotate the joint
                        mapping.unityJointTransform.localRotation =
                            Quaternion.Euler(0, unityPosition * Mathf.Rad2Deg, 0);
                        break;

                    case JointType.Prismatic:
                        // Translate the joint (along Z-axis as example)
                        Vector3 newPos = mapping.unityJointTransform.localPosition;
                        newPos.z = Mathf.Clamp(unityPosition, mapping.minAngle, mapping.maxAngle);
                        mapping.unityJointTransform.localPosition = newPos;
                        break;

                    case JointType.Fixed:
                        // No movement for fixed joints
                        break;
                }
            }
        }
    }

    // Method to send joint commands back to ROS
    public void SendJointCommands(Dictionary<string, float> targetPositions)
    {
        JointStateMsg cmd = new JointStateMsg();

        List<string> names = new List<string>();
        List<double> positions = new List<double>();

        foreach (var kvp in targetPositions)
        {
            names.Add(kvp.Key);
            positions.Add(kvp.Value);
        }

        cmd.name_array = names.ToArray();
        cmd.position_array = positions.ConvertAll(d => (float)d).ToArray();

        ROSConnection.GetOrCreateInstance().Send(jointStateTopic + "_command", cmd);
    }
}
```

### Human-Robot Interaction Scenarios

#### 1. Interactive Object System
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class InteractiveObject : MonoBehaviour
{
    [Header("Interaction Settings")]
    public string objectId;
    public bool isGrabbable = false;
    public bool isTouchable = true;
    public bool isUsable = false;

    [Header("ROS Communication")]
    public string interactionTopic = "/interaction_events";

    private bool isSelected = false;
    private Color originalColor;
    private Renderer objectRenderer;

    void Start()
    {
        objectRenderer = GetComponent<Renderer>();
        if (objectRenderer != null)
        {
            originalColor = objectRenderer.material.color;
        }
    }

    void OnMouseEnter()
    {
        if (objectRenderer != null)
        {
            objectRenderer.material.color = Color.yellow;
        }
    }

    void OnMouseExit()
    {
        if (objectRenderer != null)
        {
            objectRenderer.material.color = originalColor;
        }
    }

    void OnMouseDown()
    {
        HandleInteraction();
    }

    void HandleInteraction()
    {
        if (isGrabbable)
        {
            GrabObject();
        }
        else if (isUsable)
        {
            UseObject();
        }
        else if (isTouchable)
        {
            TouchObject();
        }

        // Send interaction event to ROS
        SendInteractionEvent();
    }

    void GrabObject()
    {
        // Implement grabbing logic
        isSelected = !isSelected;
        if (isSelected)
        {
            // Change color to indicate selection
            if (objectRenderer != null)
            {
                objectRenderer.material.color = Color.green;
            }

            // Make object follow mouse/finger
            GetComponent<Rigidbody>().isKinematic = true;
        }
        else
        {
            // Release object
            if (objectRenderer != null)
            {
                objectRenderer.material.color = originalColor;
            }

            GetComponent<Rigidbody>().isKinematic = false;
        }
    }

    void UseObject()
    {
        // Implement usage logic (e.g., turning on/off a lamp)
        Debug.Log($"Using object: {gameObject.name}");
    }

    void TouchObject()
    {
        // Implement touch feedback
        Debug.Log($"Touching object: {gameObject.name}");

        // Visual feedback
        if (objectRenderer != null)
        {
            objectRenderer.material.color = Color.blue;
            Invoke("ResetColor", 0.2f);
        }
    }

    void ResetColor()
    {
        if (objectRenderer != null)
        {
            objectRenderer.material.color = originalColor;
        }
    }

    void SendInteractionEvent()
    {
        // Create and send interaction message to ROS
        var interactionMsg = new RosMessageTypes.Std.StringMsg();
        interactionMsg.data = $"interaction:{objectId}:{Time.time}";

        ROSConnection.GetOrCreateInstance().Send(interactionTopic, interactionMsg);
    }
}
```

#### 2. Gesture Recognition System
```csharp
using UnityEngine;
using System.Collections.Generic;

public class GestureRecognizer : MonoBehaviour
{
    [Header("Gesture Settings")]
    public float gestureThreshold = 0.1f;
    public float gestureTimeout = 1.0f;

    [Header("ROS Communication")]
    public string gestureTopic = "/gesture_events";

    private List<Vector3> gesturePath = new List<Vector3>();
    private float gestureStartTime;
    private bool isTrackingGesture = false;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            StartGestureTracking();
        }
        else if (Input.GetMouseButton(0) && isTrackingGesture)
        {
            TrackGesture();
        }
        else if (Input.GetMouseButtonUp(0) && isTrackingGesture)
        {
            RecognizeAndSendGesture();
        }

        // Timeout check
        if (isTrackingGesture && Time.time - gestureStartTime > gestureTimeout)
        {
            ResetGestureTracking();
        }
    }

    void StartGestureTracking()
    {
        gesturePath.Clear();
        gestureStartTime = Time.time;
        isTrackingGesture = true;

        // Get current mouse position in world coordinates
        Vector3 worldPos = GetWorldPosition(Input.mousePosition);
        gesturePath.Add(worldPos);
    }

    void TrackGesture()
    {
        Vector3 worldPos = GetWorldPosition(Input.mousePosition);

        // Only add point if it's far enough from the last one
        if (Vector3.Distance(worldPos, gesturePath[gesturePath.Count - 1]) > gestureThreshold)
        {
            gesturePath.Add(worldPos);
        }
    }

    void RecognizeAndSendGesture()
    {
        if (gesturePath.Count < 3)
        {
            ResetGestureTracking();
            return;
        }

        // Simple gesture recognition (in a real system, you'd use more sophisticated algorithms)
        string gestureType = RecognizeGesture(gesturePath);

        // Send gesture to ROS
        SendGestureToRos(gestureType, gesturePath);

        ResetGestureTracking();
    }

    string RecognizeGesture(List<Vector3> path)
    {
        // Simple gesture recognition based on path shape
        if (path.Count < 10) return "short_gesture";

        // Calculate bounding box
        Vector3 min = path[0];
        Vector3 max = path[0];

        foreach (Vector3 point in path)
        {
            min = Vector3.Min(min, point);
            max = Vector3.Max(max, point);
        }

        Vector3 dimensions = max - min;

        // Recognize basic gestures
        if (dimensions.x > dimensions.y * 2)
        {
            return "horizontal_swipe";
        }
        else if (dimensions.y > dimensions.x * 2)
        {
            return "vertical_swipe";
        }
        else if (IsCircular(path))
        {
            return "circle";
        }
        else
        {
            return "complex_gesture";
        }
    }

    bool IsCircular(List<Vector3> path)
    {
        // Simple circle detection
        if (path.Count < 10) return false;

        Vector3 center = Vector3.zero;
        foreach (Vector3 point in path)
        {
            center += point;
        }
        center /= path.Count;

        float avgRadius = 0;
        foreach (Vector3 point in path)
        {
            avgRadius += Vector3.Distance(point, center);
        }
        avgRadius /= path.Count;

        // Check if points are roughly equidistant from center
        float tolerance = 0.2f; // 20% tolerance
        foreach (Vector3 point in path)
        {
            float distance = Vector3.Distance(point, center);
            if (Mathf.Abs(distance - avgRadius) / avgRadius > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    Vector3 GetWorldPosition(Vector3 screenPos)
    {
        // Convert screen position to world position (assuming a camera setup)
        Ray ray = Camera.main.ScreenPointToRay(screenPos);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, 100f))
        {
            return hit.point;
        }

        // Fallback: project onto a plane
        Plane plane = new Plane(Vector3.up, Vector3.zero);
        float distance;
        if (plane.Raycast(ray, out distance))
        {
            return ray.GetPoint(distance);
        }

        return Vector3.zero;
    }

    void SendGestureToRos(string gestureType, List<Vector3> path)
    {
        // In a real implementation, you would send a custom ROS message
        // For now, we'll send a string message
        var gestureMsg = new RosMessageTypes.Std.StringMsg();
        gestureMsg.data = $"gesture:{gestureType}:{path.Count}points";

        ROSConnection.GetOrCreateInstance().Send(gestureTopic, gestureMsg);

        Debug.Log($"Sent gesture: {gestureType} with {path.Count} points");
    }

    void ResetGestureTracking()
    {
        isTrackingGesture = false;
        gesturePath.Clear();
    }
}
```

### Hybrid Simulation (Unity + Gazebo)

#### 1. Unity-Gazebo Bridge Setup
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Nav;
using RosMessageTypes.Geometry;

public class UnityGazeboBridge : MonoBehaviour
{
    [Header("Topic Configuration")]
    public string unityPoseTopic = "/unity/robot_pose";
    public string gazeboCmdTopic = "/gazebo/cmd_vel";
    public string unityControlTopic = "/unity/control";

    [Header("Synchronization Settings")]
    public float syncRate = 30f; // Hz
    public bool enableForwardSync = true; // Unity -> Gazebo
    public bool enableBackwardSync = true; // Gazebo -> Unity

    private float lastSyncTime;
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to Gazebo updates
        ros.Subscribe<OdometryMsg>("/gazebo/robot/odom", ReceiveGazeboState);

        // Start synchronization
        lastSyncTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastSyncTime >= 1f / syncRate)
        {
            SyncUnityToGazebo();
            lastSyncTime = Time.time;
        }
    }

    void SyncUnityToGazebo()
    {
        if (enableForwardSync)
        {
            // Send Unity robot state to Gazebo
            var poseMsg = new PoseStampedMsg();
            poseMsg.header.frame_id = "map";
            poseMsg.header.stamp = new TimeStamp(Time.time);

            poseMsg.pose.position = new RosMessageTypes.Geometry.PointMsg(
                transform.position.x,
                transform.position.y,
                transform.position.z
            );

            poseMsg.pose.orientation = new RosMessageTypes.Geometry.QuaternionMsg(
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w
            );

            ros.Send(unityPoseTopic, poseMsg);
        }
    }

    void ReceiveGazeboState(OdometryMsg odom)
    {
        if (enableBackwardSync)
        {
            // Update Unity robot position based on Gazebo simulation
            Vector3 gazeboPosition = new Vector3(
                (float)odom.pose.pose.position.x,
                (float)odom.pose.pose.position.y,
                (float)odom.pose.pose.position.z
            );

            Quaternion gazeboRotation = new Quaternion(
                (float)odom.pose.pose.orientation.x,
                (float)odom.pose.pose.orientation.y,
                (float)odom.pose.pose.orientation.z,
                (float)odom.pose.pose.orientation.w
            );

            // Smoothly interpolate to avoid jittering
            transform.position = Vector3.Lerp(transform.position, gazeboPosition, 0.1f);
            transform.rotation = Quaternion.Slerp(transform.rotation, gazeboRotation, 0.1f);
        }
    }
}
```

## Exercises

1. Create a Unity scene with an apartment environment using the provided scripts
2. Implement a LiDAR point cloud visualization that updates in real-time from ROS
3. Create an RGB-D camera visualization system with both RGB and depth feeds
4. Develop a robot state synchronizer that keeps Unity and Gazebo in sync
5. Design interactive objects in Unity that respond to user input and send events to ROS
6. Implement a gesture recognition system for human-robot interaction
7. Create a hybrid simulation setup that synchronizes between Unity and Gazebo
8. Add physics-based interactions between the robot and environment objects

## References

1. Unity Robotics Hub: https://github.com/Unity-Technologies/Unity-Robotics-Hub
2. ROS-TCP-Connector: https://github.com/Unity-Technologies/ROS-TCP-Connector
3. URDF Importer: https://github.com/Unity-Technologies/Unity-Robotics-Hub/tree/main/submodules/URDF-Importer
4. Unity Manual: https://docs.unity3d.com/Manual/index.html

## Further Reading

- Advanced Unity shader programming for robotics visualization
- Integration with NVIDIA Isaac Sim for photorealistic simulation
- Machine learning in Unity for robot behavior training
- Multi-user collaboration in Unity for distributed robotics development
- Performance optimization for large-scale simulation environments