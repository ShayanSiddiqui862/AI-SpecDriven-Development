---
sidebar_position: 22
---

# URDF-to-SDF Conversion for High-Fidelity Simulation

## Learning Objectives
By the end of this module, students will be able to:
- Convert URDF robot models to SDF format for Gazebo simulation
- Preserve physics properties during the conversion process
- Add Gazebo-specific plugins and extensions to SDF models
- Validate converted models for simulation accuracy
- Optimize SDF models for high-fidelity simulation

## Theory

### URDF vs SDF
- **URDF (Unified Robot Description Format)**: Primarily used for ROS robot description, focusing on kinematic chains and basic physical properties
- **SDF (Simulation Description Format)**: Used by Gazebo for simulation, supporting more detailed physics properties, plugins, and simulation-specific features

### Key Differences
- **Physics Engine**: SDF supports multiple physics engines with detailed parameters
- **Plugins**: SDF has native support for Gazebo plugins
- **Sensors**: SDF provides more detailed sensor modeling
- **Materials**: SDF offers more advanced material properties
- **World Elements**: SDF can describe entire worlds, not just robots

### Conversion Considerations
- Physics properties need to be enhanced for simulation
- Additional collision and visual properties may be required
- Gazebo-specific plugins need to be added
- Joint limits and dynamics should be preserved

## Implementation

### Prerequisites
- ROS 2 Humble installed
- Gazebo Garden or higher
- Basic understanding of both URDF and SDF formats
- Experience with XML editing

### Manual Conversion Process

#### 1. Basic URDF to SDF Conversion
Start with a simple URDF and convert it step by step:

**Original URDF (simple_robot.urdf):**
```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <link name="arm_link">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>

  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
```

**Converted SDF (simple_robot.sdf):**
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <pose>0 0 0.1 0 0 0</pose>  <!-- Added pose with slight height to avoid ground collision -->

    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.02</izz>
        </inertia>
      </inertial>

      <collision name="base_collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.5</mu>
              <mu2>0.5</mu2>
            </ode>
          </friction>
          <bounce>
            <restitution_coefficient>0.01</restitution_coefficient>
            <threshold>100000</threshold>
          </bounce>
          <contact>
            <ode>
              <max_vel>100.0</max_vel>
              <min_depth>0.001</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>

      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <link name="arm_link">
      <pose>0.3 0 0 0 0 0</pose>  <!-- Pose relative to model origin -->
      <inertial>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.001</iyy>
          <iyz>0.0</iyz>
          <izz>0.0005</izz>
        </inertia>
      </inertial>

      <collision name="arm_collision">
        <geometry>
          <cylinder>
            <length>0.3</length>
            <radius>0.05</radius>
          </cylinder>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.5</mu>
              <mu2>0.5</mu2>
            </ode>
          </friction>
        </surface>
      </collision>

      <visual name="arm_visual">
        <geometry>
          <cylinder>
            <length>0.3</length>
            <radius>0.05</radius>
          </cylinder>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <joint name="arm_joint" type="revolute">
      <parent>base_link</parent>
      <child>arm_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

### Enhanced Conversion with Gazebo-Specific Features

#### 1. Adding Physics Properties
```xml
<!-- Enhanced collision properties for better simulation -->
<collision name="enhanced_collision">
  <geometry>
    <box>
      <size>0.5 0.5 0.2</size>
    </box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>0.8</mu>      <!-- Higher friction for better grip -->
        <mu2>0.8</mu2>
        <fdir1>0 0 1</fdir1>  <!-- Friction direction -->
      </ode>
      <torsional>
        <coefficient>0.1</coefficient>  <!-- Torsional friction -->
        <use_patch_radius>false</use_patch_radius>
        <surface_radius>0.01</surface_radius>
      </torsional>
    </friction>
    <bounce>
      <restitution_coefficient>0.01</restitution_coefficient>
      <threshold>100000</threshold>
    </bounce>
    <contact>
      <ode>
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
  <max_contacts>10</max_contacts>  <!-- Maximum contact points -->
</collision>
```

#### 2. Adding Sensors
```xml
<!-- Add a camera sensor to the robot -->
<link name="camera_link">
  <pose>0.2 0 0.1 0 0 0</pose>
  <sensor name="camera" type="camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <always_on>1</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
  </sensor>
</link>

<!-- Add a LiDAR sensor -->
<link name="lidar_link">
  <sensor name="laser" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>  <!-- -180 degrees -->
          <max_angle>3.14159</max_angle>    <!-- 180 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <always_on>1</always_on>
    <update_rate>10</update_rate>
    <visualize>true</visualize>
  </sensor>
</link>
```

#### 3. Adding Gazebo Plugins
```xml
<!-- Add a differential drive plugin for mobile base -->
<plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
  <ros>
    <namespace>simple_robot</namespace>
    <remapping>cmd_vel:=cmd_vel</remapping>
    <remapping>odom:=odom</remapping>
  </ros>
  <update_rate>30</update_rate>
  <left_joint>left_wheel_joint</left_joint>
  <right_joint>right_wheel_joint</right_joint>
  <wheel_separation>0.3</wheel_separation>
  <wheel_diameter>0.1</wheel_diameter>
  <max_wheel_torque>20</max_wheel_torque>
  <max_wheel_acceleration>1.0</max_wheel_acceleration>
  <publish_odom>true</publish_odom>
  <publish_odom_tf>true</publish_odom_tf>
  <publish_wheel_tf>true</publish_wheel_tf>
  <odometry_frame>odom</odometry_frame>
  <robot_base_frame>base_link</robot_base_frame>
</plugin>

<!-- Add joint state publisher -->
<plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
  <ros>
    <namespace>simple_robot</namespace>
    <remapping>joint_states:=joint_states</remapping>
  </ros>
  <update_rate>30</update_rate>
  <joint_name>arm_joint</joint_name>
</plugin>
```

### Automated Conversion Tools

#### 1. Using xacro to SDF conversion
For robots defined with xacro, you can convert to SDF using:

```bash
# First convert xacro to URDF
ros2 run xacro xacro input_robot.xacro > output_robot.urdf

# Then manually convert to SDF or use tools
```

#### 2. Python script for conversion assistance
Create `urdf_to_sdf_converter.py`:

```python
#!/usr/bin/env python3

"""
Utility script to help convert URDF to SDF format.
This script provides a basic framework that can be extended for specific needs.
"""

import xml.etree.ElementTree as ET
import argparse


def urdf_to_sdf(urdf_content):
    """
    Convert URDF XML content to basic SDF structure.
    This is a simplified converter - real-world conversion may need more sophistication.
    """
    urdf_root = ET.fromstring(urdf_content)

    # Create SDF root
    sdf_root = ET.Element("sdf", version="1.7")
    model = ET.SubElement(sdf_root, "model", name=urdf_root.get("name", "converted_model"))

    # Process links
    for link in urdf_root.findall("link"):
        sdf_link = convert_link(link)
        model.append(sdf_link)

    # Process joints
    for joint in urdf_root.findall("joint"):
        sdf_joint = convert_joint(joint)
        model.append(sdf_joint)

    return ET.tostring(sdf_root, encoding="unicode")


def convert_link(urdf_link):
    """Convert a URDF link to SDF format."""
    sdf_link = ET.Element("link", name=urdf_link.get("name"))

    # Copy pose (if not specified in URDF, use default)
    pose_elem = ET.SubElement(sdf_link, "pose")
    pose_elem.text = "0 0 0 0 0 0"

    # Convert inertial properties
    urdf_inertial = urdf_link.find("inertial")
    if urdf_inertial is not None:
        sdf_inertial = ET.SubElement(sdf_link, "inertial")

        mass = urdf_inertial.find("mass")
        if mass is not None:
            mass_elem = ET.SubElement(sdf_inertial, "mass")
            mass_elem.text = mass.get("value", "0.1")

        inertia = urdf_inertial.find("inertia")
        if inertia is not None:
            sdf_inertia = ET.SubElement(sdf_inertial, "inertia")
            for attr in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
                val = inertia.get(f"{attr}", "0.001")
                elem = ET.SubElement(sdf_inertia, attr)
                elem.text = val

    # Convert collision properties
    urdf_collision = urdf_link.find("collision")
    if urdf_collision is not None:
        sdf_collision = ET.SubElement(sdf_link, "collision", name=f"{urdf_link.get('name')}_collision")

        # Geometry
        urdf_geom = urdf_collision.find("geometry")
        if urdf_geom is not None:
            sdf_geom = ET.SubElement(sdf_collision, "geometry")
            copy_geometry(urdf_geom, sdf_geom)

        # Surface properties for simulation
        surface = ET.SubElement(sdf_collision, "surface")
        friction = ET.SubElement(surface, "friction")
        ode = ET.SubElement(friction, "ode")
        mu = ET.SubElement(ode, "mu")
        mu.text = "0.5"
        mu2 = ET.SubElement(ode, "mu2")
        mu2.text = "0.5"

    # Convert visual properties
    urdf_visual = urdf_link.find("visual")
    if urdf_visual is not None:
        sdf_visual = ET.SubElement(sdf_link, "visual", name=f"{urdf_link.get('name')}_visual")

        # Geometry
        urdf_geom = urdf_visual.find("geometry")
        if urdf_geom is not None:
            sdf_geom = ET.SubElement(sdf_visual, "geometry")
            copy_geometry(urdf_geom, sdf_geom)

        # Material
        urdf_material = urdf_visual.find("material")
        if urdf_material is not None:
            sdf_material = ET.SubElement(sdf_visual, "material")
            # Add basic material properties

    return sdf_link


def convert_joint(urdf_joint):
    """Convert a URDF joint to SDF format."""
    sdf_joint = ET.Element("joint", name=urdf_joint.get("name"),
                          type=urdf_joint.get("type", "revolute"))

    # Pose
    pose_elem = ET.SubElement(sdf_joint, "pose")
    pose_elem.text = "0 0 0 0 0 0"

    # Parent and child
    parent = ET.SubElement(sdf_joint, "parent")
    parent.text = urdf_joint.find("parent").get("link")

    child = ET.SubElement(sdf_joint, "child")
    child.text = urdf_joint.find("child").get("link")

    # Axis
    axis = urdf_joint.find("axis")
    if axis is not None:
        sdf_axis = ET.SubElement(sdf_joint, "axis")
        xyz = ET.SubElement(sdf_axis, "xyz")
        xyz.text = axis.get("xyz", "0 0 1")

    # Limits
    limit = urdf_joint.find("limit")
    if limit is not None:
        sdf_axis = sdf_joint.find("axis")
        if sdf_axis is None:
            sdf_axis = ET.SubElement(sdf_joint, "axis")

        sdf_limit = ET.SubElement(sdf_axis, "limit")
        lower = ET.SubElement(sdf_limit, "lower")
        lower.text = limit.get("lower", "-1e16")
        upper = ET.SubElement(sdf_limit, "upper")
        upper.text = limit.get("upper", "1e16")
        effort = ET.SubElement(sdf_limit, "effort")
        effort.text = limit.get("effort", "1e16")
        velocity = ET.SubElement(sdf_limit, "velocity")
        velocity.text = limit.get("velocity", "1e16")

    return sdf_joint


def copy_geometry(urdf_geom, sdf_geom):
    """Copy geometry from URDF to SDF format."""
    for child in urdf_geom:
        sdf_child = ET.SubElement(sdf_geom, child.tag)
        if child.tag == "box":
            size = ET.SubElement(sdf_child, "size")
            size.text = child.get("size", "0.1 0.1 0.1")
        elif child.tag == "cylinder":
            radius = ET.SubElement(sdf_child, "radius")
            radius.text = child.get("radius", "0.05")
            length = ET.SubElement(sdf_child, "length")
            length.text = child.get("length", "0.1")
        elif child.tag == "sphere":
            radius = ET.SubElement(sdf_child, "radius")
            radius.text = child.get("radius", "0.05")


def main():
    parser = argparse.ArgumentParser(description="Convert URDF to SDF")
    parser.add_argument("input", help="Input URDF file")
    parser.add_argument("output", help="Output SDF file")

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        urdf_content = f.read()

    sdf_content = urdf_to_sdf(urdf_content)

    with open(args.output, 'w') as f:
        f.write(sdf_content)

    print(f"Converted {args.input} to {args.output}")


if __name__ == "__main__":
    main()
```

### Best Practices for High-Fidelity Simulation

#### 1. Physics Tuning
```xml
<!-- Enhanced physics properties for realistic simulation -->
<physics type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>1e-5</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

#### 2. Collision Mesh Optimization
```xml
<!-- Use simplified collision meshes for better performance -->
<collision name="simplified_collision">
  <geometry>
    <!-- Instead of complex mesh, use simpler primitives -->
    <box>
      <size>0.5 0.3 0.2</size>
    </box>
  </geometry>
  <!-- Add multiple simple shapes if needed -->
  <surface>
    <friction>
      <ode>
        <mu>0.5</mu>
        <mu2>0.5</mu2>
      </ode>
    </friction>
  </surface>
</collision>

<visual name="detailed_visual">
  <geometry>
    <!-- Use detailed mesh for visual appearance -->
    <mesh>
      <uri>model://robot/meshes/detailed_body.dae</uri>
    </mesh>
  </geometry>
</visual>
```

#### 3. Inertial Property Verification
```xml
<!-- Properly calculated inertial properties are crucial for realistic simulation -->
<inertial>
  <mass>1.0</mass>
  <inertia>
    <ixx>0.01</ixx>
    <ixy>0.0</ixy>  <!-- Should be 0 for symmetric objects -->
    <ixz>0.0</ixz>  <!-- Should be 0 for symmetric objects -->
    <iyy>0.01</iyy>
    <iyz>0.0</iyz>  <!-- Should be 0 for symmetric objects -->
    <izz>0.02</izz>
  </inertia>
</inertial>
```

### Validation and Testing

#### 1. SDF Validation
Before using the converted SDF in Gazebo, validate it:

```bash
# Use gz command to check the model
gz sdf -k model.sdf

# Or use check sdf
gz sdf -p model.sdf
```

#### 2. Loading in Gazebo
```bash
# Test the model in Gazebo
gazebo -s libgazebo_ros_factory.so
# Then spawn the model using:
# ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/model.sdf
```

#### 3. Physics Behavior Testing
- Check if the robot falls through the ground
- Verify joint movements are realistic
- Test collision detection
- Validate sensor outputs

## Exercises

1. Convert the simple URDF robot example to SDF format manually
2. Add physics properties to make the simulation more realistic
3. Include a camera and LiDAR sensor in your SDF model
4. Add ROS 2 Gazebo plugins to enable control and feedback
5. Validate your converted model in Gazebo
6. Create a complete humanoid robot model in SDF format
7. Optimize collision meshes for better simulation performance

## References

1. SDF Specification: http://sdformat.org/
2. URDF to SDF Conversion Guide: http://gazebosim.org/tutorials?tut=ros_urdf
3. Gazebo Model Tutorial: http://gazebosim.org/tutorials?tut=models
4. Physics Parameters Guide: http://gazebosim.org/tutorials/?tut=physics_ros

## Further Reading

- Advanced SDF features and extensions
- Custom Gazebo plugins for specialized simulation
- Integration with ROS 2 control systems
- Performance optimization techniques for large models