---
sidebar_position: 21
---

# Gazebo World Building with Physics Properties

## Learning Objectives
By the end of this module, students will be able to:
- Create SDF world files for Gazebo simulation
- Configure physics properties for realistic simulation
- Design apartment environments with interactive objects
- Set up lighting and visual properties for realistic rendering
- Optimize world files for performance

## Theory

### SDF (Simulation Description Format)
SDF is an XML-based format that describes environments, objects, and robots in Gazebo. It defines:
- World properties (gravity, physics engine)
- Models (robots, furniture, objects)
- Lighting and visual effects
- Physics properties (friction, damping, restitution)

### Key Physics Concepts
- **Gravity**: Force that pulls objects toward the ground
- **Friction**: Resistance to sliding motion between surfaces
- **Restitution**: Bounciness of collisions (0 = no bounce, 1 = perfectly elastic)
- **Damping**: Energy loss that reduces motion over time
- **Solver**: Algorithm that calculates physics interactions

### World Structure
A Gazebo world consists of:
- World definition with global properties
- Models that represent objects in the environment
- Plugins that add functionality
- Lighting and visual effects

## Implementation

### Prerequisites
- Gazebo Garden or higher installed
- Basic understanding of XML
- Knowledge of physics concepts

### Basic World File Structure

#### 1. Creating a Basic World File
Create `basic_apartment.world`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="apartment_world">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics Engine Configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.0</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Apartment Room -->
    <model name="apartment_room">
      <pose>0 0 0 0 0 0</pose>
      <static>true</static>

      <!-- Room walls -->
      <link name="room_walls">
        <collision name="room_collision">
          <geometry>
            <box>
              <size>6 6 3</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
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

        <visual name="room_visual">
          <geometry>
            <box>
              <size>6 6 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture: Table -->
    <model name="table">
      <pose>1 0 0 0 0 0</pose>
      <link name="table_top">
        <collision name="table_collision">
          <geometry>
            <box>
              <size>1.2 0.6 0.02</size>
            </box>
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
        <visual name="table_visual">
          <geometry>
            <box>
              <size>1.2 0.6 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
      </link>

      <link name="leg1">
        <pose>-0.5 -0.25 0 0 0 0</pose>
        <collision name="leg1_collision">
          <geometry>
            <cylinder>
              <radius>0.03</radius>
              <length>0.7</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="leg1_visual">
          <geometry>
            <cylinder>
              <radius>0.03</radius>
              <length>0.7</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.4 0.2 0.1 1</ambient>
            <diffuse>0.4 0.2 0.1 1</diffuse>
          </material>
        </visual>
      </link>

      <link name="leg2">
        <pose>0.5 -0.25 0 0 0 0</pose>
        <collision name="leg2_collision">
          <geometry>
            <cylinder>
              <radius>0.03</radius>
              <length>0.7</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="leg2_visual">
          <geometry>
            <cylinder>
              <radius>0.03</radius>
              <length>0.7</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.4 0.2 0.1 1</ambient>
            <diffuse>0.4 0.2 0.1 1</diffuse>
          </material>
        </visual>
      </link>

      <joint name="top_to_leg1" type="fixed">
        <parent>table_top</parent>
        <child>leg1</child>
      </joint>

      <joint name="top_to_leg2" type="fixed">
        <parent>table_top</parent>
        <child>leg2</child>
      </joint>
    </model>

    <!-- Furniture: Chair -->
    <model name="chair">
      <pose>1.5 0.8 0 0 0 1.57</pose>
      <link name="chair_seat">
        <collision name="seat_collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="seat_visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
          </material>
        </visual>
      </link>

      <link name="chair_back">
        <pose>0 0 0.2 0 0 0</pose>
        <collision name="back_collision">
          <geometry>
            <box>
              <size>0.4 0.02 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="back_visual">
          <geometry>
            <box>
              <size>0.4 0.02 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
          </material>
        </visual>
      </link>

      <link name="leg1">
        <pose>-0.15 -0.15 0 0 0 0</pose>
        <collision name="leg1_collision">
          <geometry>
            <cylinder>
              <radius>0.02</radius>
              <length>0.45</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="leg1_visual">
          <geometry>
            <cylinder>
              <radius>0.02</radius>
              <length>0.45</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.3 0.15 0.05 1</ambient>
            <diffuse>0.3 0.15 0.05 1</diffuse>
          </material>
        </visual>
      </link>

      <joint name="seat_to_back" type="fixed">
        <parent>chair_seat</parent>
        <child>chair_back</child>
      </joint>

      <joint name="seat_to_leg1" type="fixed">
        <parent>chair_seat</parent>
        <child>leg1</child>
      </joint>
    </model>
  </world>
</sdf>
```

### Advanced Physics Configuration

#### 1. Physics Engine Tuning
For realistic simulation, adjust physics parameters based on your needs:

```xml
<physics type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>  <!-- Increase for stability -->
      <sor>1.3</sor>      <!-- Successive over-relaxation -->
    </solver>
    <constraints>
      <cfm>0.000001</cfm>    <!-- Constraint Force Mixing -->
      <erp>0.2</erp>        <!-- Error Reduction Parameter -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

#### 2. Material Properties for Realistic Simulation

```xml
<material>
  <script>
    <uri>file://media/materials/scripts/gazebo.material</uri>
    <name>Gazebo/White</name>
  </script>
  <ambient>0.8 0.8 0.8 1</ambient>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.1 0.1 0.1 1</specular>
  <emissive>0 0 0 1</emissive>
</material>
```

### Creating Apartment-Specific Elements

#### 1. Room Layout
Create `apartment_layout.sdf` with multiple rooms:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="apartment_layout">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Living Room -->
    <model name="living_room">
      <pose>0 0 0 0 0 0</pose>
      <static>true</static>
      <link name="living_room_structure">
        <collision name="living_room_collision">
          <geometry>
            <box>
              <size>5 4 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="living_room_visual">
          <geometry>
            <box>
              <size>5 4 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Kitchen -->
    <model name="kitchen">
      <pose>0 -3 0 0 0 0</pose>
      <static>true</static>
      <link name="kitchen_structure">
        <collision name="kitchen_collision">
          <geometry>
            <box>
              <size>3 3 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="kitchen_visual">
          <geometry>
            <box>
              <size>3 3 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.85 0.85 0.85 1</ambient>
            <diffuse>0.85 0.85 0.85 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Bedroom -->
    <model name="bedroom">
      <pose>3 0 0 0 0 0</pose>
      <static>true</static>
      <link name="bedroom_structure">
        <collision name="bedroom_collision">
          <geometry>
            <box>
              <size>4 4 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="bedroom_visual">
          <geometry>
            <box>
              <size>4 4 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.88 0.88 0.88 1</ambient>
            <diffuse>0.88 0.88 0.88 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture models can be included here -->
    <include>
      <uri>model://sofa</uri>
      <pose>1 0.5 0 0 0 0</pose>
    </include>

    <include>
      <uri>model://kitchen_table</uri>
      <pose>0 -1.5 0 0 0 0</pose>
    </include>

    <include>
      <uri>model://bed</uri>
      <pose>4 0 0 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Physics Properties for Realistic Simulation

#### 1. Friction Settings
Different materials require different friction coefficients:

```xml
<!-- High friction (carpet, rubber) -->
<surface>
  <friction>
    <ode>
      <mu>0.8</mu>
      <mu2>0.8</mu2>
    </ode>
  </friction>
</surface>

<!-- Medium friction (wood, tile) -->
<surface>
  <friction>
    <ode>
      <mu>0.5</mu>
      <mu2>0.5</mu2>
    </ode>
  </friction>
</surface>

<!-- Low friction (ice, polished surface) -->
<surface>
  <friction>
    <ode>
      <mu>0.1</mu>
      <mu2>0.1</mu2>
    </ode>
  </friction>
</surface>
```

#### 2. Restitution (Bounciness)
```xml
<!-- Non-bouncy (carpet, soft materials) -->
<surface>
  <bounce>
    <restitution_coefficient>0.01</restitution_coefficient>
  </bounce>
</surface>

<!-- Slightly bouncy (wood, plastic) -->
<surface>
  <bounce>
    <restitution_coefficient>0.1</restitution_coefficient>
  </bounce>
</surface>

<!-- Very bouncy (rubber ball) -->
<surface>
  <bounce>
    <restitution_coefficient>0.8</restitution_coefficient>
  </bounce>
</surface>
```

### Lighting and Visual Properties

#### 1. Indoor Lighting Configuration
```xml
<!-- Add lighting to your world -->
<light name="room_light_1" type="point">
  <pose>0 0 2 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.8 0.8 0.8 1</specular>
  <attenuation>
    <range>10</range>
    <constant>0.5</constant>
    <linear>0.1</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
  <cast_shadows>true</cast_shadows>
</light>

<light name="kitchen_light" type="point">
  <pose>0 -1.5 2 0 0 0</pose>
  <diffuse>1 0.9 0.8 1</diffuse>
  <specular>1 0.9 0.8 1</specular>
  <attenuation>
    <range>8</range>
    <constant>0.5</constant>
    <linear>0.1</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
</light>
```

### World Optimization Techniques

#### 1. Performance Optimization
```xml
<!-- Use simpler collision geometries for complex objects -->
<collision name="simplified_collision">
  <geometry>
    <!-- Use a box instead of complex mesh for collision -->
    <box>
      <size>1 1 1</size>
    </box>
  </geometry>
</collision>

<visual name="detailed_visual">
  <geometry>
    <!-- Use detailed mesh for visual appearance -->
    <mesh>
      <uri>model://my_object/meshes/detailed_model.dae</uri>
    </mesh>
  </geometry>
</visual>
```

#### 2. Static vs Dynamic Objects
```xml
<!-- Use static for unmovable objects like walls -->
<model name="wall">
  <static>true</static>
  <!-- ... -->
</model>

<!-- Omit static tag for movable objects -->
<model name="chair">
  <!-- This will be dynamic by default -->
  <!-- ... -->
</model>
```

### Creating Custom Models

#### 1. Model Directory Structure
```
models/
└── my_apartment_furniture/
    ├── model.sdf
    ├── model.config
    └── meshes/
        ├── chair.dae
        └── table.dae
```

#### 2. Model Configuration File (model.config)
```xml
<?xml version="1.0"?>
<model>
  <name>apartment_furniture</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A set of apartment furniture for Gazebo simulation.</description>
</model>
```

## Exercises

1. Create a basic apartment world with at least 3 rooms and 5 pieces of furniture
2. Configure physics properties to make the simulation stable and realistic
3. Add proper lighting to your apartment world
4. Create custom models for specific furniture pieces
5. Test the world by spawning a robot and navigating through it
6. Optimize your world file for better performance
7. Add interactive objects that respond to robot contact

## References

1. Gazebo SDF Documentation: http://sdformat.org/
2. Gazebo World Tutorial: http://gazebosim.org/tutorials?tut=build_world
3. Physics Parameters Guide: http://gazebosim.org/tutorials/?tut=physics_ros
4. Model Creation Tutorial: http://gazebosim.org/tutorials?tut=build_model

## Further Reading

- Advanced physics tuning for specific robot types
- Procedural world generation techniques
- Multi-robot simulation in the same environment
- Integration with ROS 2 for dynamic world modification