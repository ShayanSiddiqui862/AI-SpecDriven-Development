---
sidebar_position: 41
---

# LLM Prompt Engineering for Task Decomposition in Robotics

## Learning Objectives
By the end of this module, students will be able to:
- Design effective prompts for LLM-based task decomposition in robotics
- Implement context-aware prompting for robot command interpretation
- Create structured output formats for robot action sequences
- Integrate LLMs with ROS 2 for natural language command processing
- Optimize prompt engineering for real-time robotics applications

## Theory

### Large Language Models in Robotics

Large Language Models (LLMs) are increasingly being used in robotics for:
- Natural language command interpretation
- Task planning and decomposition
- High-level behavioral reasoning
- Human-robot interaction facilitation

### Key Concepts

#### 1. Task Decomposition
Breaking down complex natural language commands into executable robot actions:
- **Sequential tasks**: Actions that must be performed in order
- **Parallel tasks**: Actions that can be performed simultaneously
- **Conditional tasks**: Actions that depend on environmental conditions
- **Iterative tasks**: Actions that repeat until a condition is met

#### 2. Prompt Engineering Fundamentals
- **Context**: Information provided to the LLM to understand the situation
- **Instruction**: Clear directive about what the LLM should do
- **Examples**: Demonstrations of desired behavior (few-shot learning)
- **Constraints**: Limitations on the output format or content

#### 3. Robotics-Specific Considerations
- **Action Space**: Limited set of available robot actions
- **Environmental Constraints**: Physical limitations of the environment
- **Safety Requirements**: Ensuring commands are safe to execute
- **Temporal Constraints**: Real-time processing requirements

### Prompt Engineering Techniques for Robotics

#### 1. Chain-of-Thought Reasoning
Breaking complex tasks into intermediate reasoning steps before producing the final action sequence.

#### 2. Few-Shot Learning
Providing examples of command-to-action mappings to guide the LLM.

#### 3. Role-Based Prompting
Defining the LLM's role as a "robotic task planner" with specific responsibilities.

## Implementation

### Prerequisites
- Access to an LLM API (OpenAI GPT, Claude, or open-source alternative)
- ROS 2 Humble with Python support
- Basic understanding of natural language processing

### 1. Basic Prompt Structure for Robotics

#### Fundamental Prompt Template
```python
# basic_robotics_prompt.py
SYSTEM_PROMPT = """
You are an AI assistant specialized in robotics task planning and decomposition. Your role is to interpret natural language commands and decompose them into structured action sequences for a humanoid robot operating in an apartment environment.

Robot Capabilities:
- Navigation: Move to specific locations (x, y, z coordinates)
- Manipulation: Pick up, place, grasp, release objects
- Perception: Detect, identify, and locate objects
- Interaction: Open/close doors, press buttons, operate switches
- Communication: Speak, listen, and respond to humans

Action Format:
Each action should be in the format: ACTION_NAME(PARAMETER1=value, PARAMETER2=value, ...)
- Navigate(location="kitchen_table")
- Detect(object="water_bottle")
- Grasp(object="water_bottle", position="above")
- Transport(from="kitchen_counter", to="dining_table")
- Place(object="water_bottle", location="dining_table")
- Speak(text="I have placed the water bottle on the dining table")

Environmental Context:
- The robot operates in a standard apartment with rooms: living room, kitchen, bedroom, bathroom
- Common objects include: chairs, tables, beds, cabinets, appliances, food items
- The robot has arms, legs, and a head with cameras and microphones

Output Requirements:
- Provide step-by-step action sequence
- Include reasoning for each step
- Consider safety and feasibility
- Use only available robot actions
"""

EXAMPLE_COMMAND = """
Command: "Please bring me a glass of water from the kitchen and place it on the coffee table in the living room."

Decomposed Actions:
1. Navigate(location="kitchen")
2. Detect(object="glass")
3. Grasp(object="glass", position="handle")
4. Detect(object="water_tap")
5. Operate(object="water_tap", action="turn_on")
6. Wait(duration=2.0)
7. Operate(object="water_tap", action="turn_off")
8. Navigate(location="living_room")
9. Detect(location="coffee_table")
10. Place(object="glass", location="coffee_table")
11. Speak(text="I have brought you a glass of water and placed it on the coffee table in the living room.")
"""
```

#### 2. Advanced Prompt with Context Awareness
```python
# context_aware_prompt.py
import json
from typing import Dict, List, Any

class RoboticsPromptEngineer:
    """
    Class for creating context-aware prompts for robotics applications.
    """

    def __init__(self):
        self.system_context = {
            "robot_capabilities": [
                "navigation_to_position",
                "object_detection",
                "object_grasping",
                "object_transportation",
                "object_placement",
                "speech_synthesis",
                "door_opening",
                "switch_operation",
                "human_interaction"
            ],
            "environment_map": {
                "rooms": ["living_room", "kitchen", "bedroom", "bathroom"],
                "common_objects": [
                    "chair", "table", "couch", "bed", "refrigerator",
                    "microwave", "cup", "plate", "bottle", "book"
                ],
                "navigable_locations": [
                    "kitchen_counter", "dining_table", "coffee_table",
                    "bedside_table", "couch_side_table"
                ]
            }
        }

    def create_task_decomposition_prompt(self,
                                       command: str,
                                       robot_state: Dict[str, Any],
                                       environment_state: Dict[str, Any]) -> str:
        """
        Create a prompt for task decomposition with current context.

        Args:
            command: Natural language command from user
            robot_state: Current state of the robot (position, battery, etc.)
            environment_state: Current state of environment (object locations, etc.)

        Returns:
            Formatted prompt string
        """
        prompt = f"""{self.get_system_instruction()}

Current Robot State:
{json.dumps(robot_state, indent=2)}

Current Environment State:
{json.dumps(environment_state, indent=2)}

Natural Language Command:
"{command}"

Please decompose this command into a sequence of executable robot actions. Follow these requirements:

1. Provide reasoning for each action step
2. Consider the current robot and environment state
3. Ensure actions are physically possible given robot capabilities
4. Output actions in the following JSON format:

{{
    "reasoning": "Explain your thought process for decomposing the task",
    "action_sequence": [
        {{
            "step": 1,
            "action": "action_name",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "reason": "Why this action is needed"
        }},
        {{
            "step": 2,
            "action": "action_name",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "reason": "Why this action is needed"
        }}
    ],
    "estimated_completion_time": "Estimated time in seconds",
    "potential_challenges": ["List any potential challenges"]
}}

Be specific about locations, objects, and parameters. Ensure each action is executable by the robot."""

        return prompt

    def get_system_instruction(self) -> str:
        """Get the system instruction part of the prompt."""
        return f"""You are an AI robotic task planner that specializes in decomposing natural language commands into executable action sequences for a humanoid robot.

Robot Capabilities:
{json.dumps(self.system_context["robot_capabilities"], indent=2)}

Environment Context:
{json.dumps(self.system_context["environment_map"], indent=2)}

Action Constraints:
- Only use actions from the capabilities list
- All locations must exist in the environment map
- Consider robot's current state and position
- Ensure actions are safe and feasible
- Account for object availability and accessibility

Output Format:
Strictly follow the JSON format specified in the request with reasoning, action sequence, estimated time, and challenges."""

    def validate_action_sequence(self, action_sequence: List[Dict]) -> bool:
        """
        Validate that the action sequence is valid for the robot.

        Args:
            action_sequence: List of action dictionaries

        Returns:
            True if valid, False otherwise
        """
        for action in action_sequence:
            if "action" not in action:
                return False

            # Check if action is in robot capabilities
            if action["action"] not in self.system_context["robot_capabilities"]:
                # Check for common action aliases
                if not self.is_valid_robot_action(action["action"]):
                    return False

        return True

    def is_valid_robot_action(self, action_name: str) -> bool:
        """Check if action name is valid (including common variations)."""
        valid_actions = set(self.system_context["robot_capabilities"])

        # Add common aliases
        action_aliases = {
            "navigate": "navigation_to_position",
            "move_to": "navigation_to_position",
            "go_to": "navigation_to_position",
            "detect": "object_detection",
            "find": "object_detection",
            "locate": "object_detection",
            "grasp": "object_grasping",
            "pick_up": "object_grasping",
            "take": "object_grasping",
            "transport": "object_transportation",
            "carry": "object_transportation",
            "move_object": "object_transportation",
            "place": "object_placement",
            "put_down": "object_placement",
            "set_down": "object_placement",
            "speak": "speech_synthesis",
            "say": "speech_synthesis",
            "talk": "speech_synthesis",
            "open_door": "door_opening",
            "close_door": "door_opening",
            "operate_switch": "switch_operation",
            "press_button": "switch_operation"
        }

        canonical_action = action_aliases.get(action_name.lower(), action_name)
        return canonical_action in valid_actions


class LLMPromptNode:
    """
    ROS 2 node that interfaces with LLM for task decomposition.
    """

    def __init__(self):
        # This would normally be a ROS 2 node, simplified for this example
        self.prompt_engineer = RoboticsPromptEngineer()
        self.llm_client = self.initialize_llm_client()  # e.g., OpenAI client

    def initialize_llm_client(self):
        """Initialize LLM client (implementation depends on chosen LLM)."""
        # Placeholder - would implement actual LLM client initialization
        pass

    def process_command(self, command: str, robot_state: Dict, env_state: Dict) -> List[Dict]:
        """
        Process a natural language command and return action sequence.

        Args:
            command: Natural language command
            robot_state: Current robot state
            env_state: Current environment state

        Returns:
            List of action dictionaries
        """
        # Create prompt with context
        prompt = self.prompt_engineer.create_task_decomposition_prompt(
            command, robot_state, env_state
        )

        # Call LLM
        response = self.call_llm(prompt)

        # Parse response
        parsed_response = self.parse_llm_response(response)

        # Validate action sequence
        if self.prompt_engineer.validate_action_sequence(parsed_response["action_sequence"]):
            return parsed_response["action_sequence"]
        else:
            raise ValueError("Invalid action sequence generated by LLM")

    def call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        # This would make an actual API call to the LLM
        # For this example, we'll return a mock response
        return self.mock_llm_response(prompt)

    def mock_llm_response(self, prompt: str) -> str:
        """Mock LLM response for demonstration."""
        return '''{
    "reasoning": "The user wants a glass of water from the kitchen placed on the coffee table in the living room. I need to navigate to the kitchen, find a glass, fill it with water, then navigate to the living room and place it on the coffee table.",
    "action_sequence": [
        {
            "step": 1,
            "action": "navigation_to_position",
            "parameters": {"location": "kitchen"},
            "reason": "Need to go to kitchen to find glass and water source"
        },
        {
            "step": 2,
            "action": "object_detection",
            "parameters": {"object_type": "glass"},
            "reason": "Look for a glass to use"
        },
        {
            "step": 3,
            "action": "object_grasping",
            "parameters": {"object": "glass", "position": "handle"},
            "reason": "Pick up the glass"
        },
        {
            "step": 4,
            "action": "object_detection",
            "parameters": {"object_type": "water_source"},
            "reason": "Find the water source (tap/filtered water dispenser)"
        },
        {
            "step": 5,
            "action": "object_transportation",
            "parameters": {"object": "glass", "destination": "water_source"},
            "reason": "Move glass to water source"
        },
        {
            "step": 6,
            "action": "switch_operation",
            "parameters": {"object": "water_source", "operation": "fill", "duration": 3.0},
            "reason": "Fill the glass with water"
        },
        {
            "step": 7,
            "action": "navigation_to_position",
            "parameters": {"location": "living_room"},
            "reason": "Navigate to living room to place the glass"
        },
        {
            "step": 8,
            "action": "object_detection",
            "parameters": {"object_type": "coffee_table"},
            "reason": "Locate the coffee table"
        },
        {
            "step": 9,
            "action": "object_placement",
            "parameters": {"object": "glass", "location": "coffee_table", "orientation": "upright"},
            "reason": "Place the water-filled glass on the coffee table"
        },
        {
            "step": 10,
            "action": "speech_synthesis",
            "parameters": {"text": "I have brought you a glass of water and placed it on the coffee table in the living room."},
            "reason": "Inform the user that the task is complete"
        }
    ],
    "estimated_completion_time": 120,
    "potential_challenges": ["Glass might not be clean", "Coffee table might be occupied", "Water source might be empty"]
}'''

    def parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response into structured format."""
        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError:
            # Handle malformed JSON response
            raise ValueError(f"LLM returned malformed JSON: {response}")


# Example usage
if __name__ == "__main__":
    engineer = RoboticsPromptEngineer()

    # Example robot and environment state
    robot_state = {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "battery_level": 0.85,
        "gripper_status": "free",
        "current_task": "idle"
    }

    environment_state = {
        "kitchen": {
            "objects": ["refrigerator", "microwave", "counter", "cabinet"],
            "accessible": True
        },
        "living_room": {
            "objects": ["sofa", "coffee_table", "tv", "couch"],
            "accessible": True
        },
        "object_locations": {
            "glass_1": {"room": "kitchen", "location": "counter"},
            "water_bottle": {"room": "kitchen", "location": "refrigerator"}
        }
    }

    command = "Bring me a glass of water from the kitchen and place it on the coffee table in the living room"

    prompt = engineer.create_task_decomposition_prompt(
        command, robot_state, environment_state
    )

    print("Generated Prompt:")
    print(prompt)
```

### 3. Specialized Prompts for Different Robot Functions

#### Navigation-Specific Prompt
```python
# navigation_prompt.py
NAVIGATION_SYSTEM_PROMPT = """
You are a navigation planning specialist for a humanoid robot. Your task is to create safe and efficient navigation paths considering the robot's bipedal nature and environmental constraints.

Robot Navigation Constraints:
- Maximum step height: 0.1m
- Maximum step length: 0.3m
- Turning radius: 0.2m (minimum)
- Climbing capability: Ramps only, no stairs
- Doorway width: Minimum 0.6m clearance
- Corridor width: Minimum 0.8m for comfortable passage

Environmental Information:
- Known obstacles and their locations
- Passable doorways and corridors
- Elevator locations (if multi-story)
- Charging stations
- Restricted areas

Navigation Actions:
- navigate_to(location, avoid_obstacles=True)
- follow_path(waypoints)
- avoid_obstacle(obstacle_id, alternative_route)
- wait_for_clear_path(duration=5.0)

Safety Considerations:
- Maintain minimum distance from obstacles (0.3m)
- Avoid areas with low ceiling (<2.0m)
- Consider dynamic obstacles (moving people/pets)
- Plan for contingencies (blocked paths)

Output Format:
Provide a detailed navigation plan with:
1. Primary path as sequence of waypoints
2. Potential obstacles and mitigation strategies
3. Estimated travel time
4. Safety considerations
5. Alternative routes if primary path is blocked
"""

def create_navigation_prompt(start_location: str, goal_location: str,
                           environment_map: Dict, robot_constraints: Dict) -> str:
    """Create navigation-specific prompt."""
    return f"""{NAVIGATION_SYSTEM_PROMPT}

Current Location: {start_location}
Destination: {goal_location}

Environment Map:
{json.dumps(environment_map, indent=2)}

Robot Constraints:
{json.dumps(robot_constraints, indent=2)}

Create a detailed navigation plan from current location to destination that accounts for the robot's bipedal constraints and environmental obstacles."""
```

#### Manipulation-Specific Prompt
```python
# manipulation_prompt.py
MANIPULATION_SYSTEM_PROMPT = """
You are a robotic manipulation planning specialist. Plan safe and effective manipulation sequences for a humanoid robot with dual arms.

Robot Manipulation Capabilities:
- Reachable workspace: 0.1m to 1.2m from base
- Gripper payload: Maximum 2kg
- Grasp types: Pinch, power, lateral
- Joint limits: As per humanoid kinematic model
- Force limits: 50N maximum endpoint force

Object Properties to Consider:
- Size, weight, shape, fragility
- Surface properties (smooth, textured, slippery)
- Center of mass and balance
- Attachment points (handles, grips)

Manipulation Actions:
- detect_object(object_id)
- approach_object(object_id, approach_vector)
- grasp_object(object_id, grasp_type, grasp_pose)
- transport_object(object_id, destination)
- place_object(object_id, destination, orientation)
- release_object(object_id)

Safety Considerations:
- Avoid collisions with environment
- Maintain robot balance during manipulation
- Respect object fragility
- Consider human safety in shared spaces

Output Format:
1. Object approach strategy
2. Grasp planning with specific pose
3. Transport path avoiding collisions
4. Placement strategy
5. Force control parameters
6. Backup plans for failed grasps
"""

def create_manipulation_prompt(task_description: str, object_properties: Dict,
                             robot_state: Dict, environment_state: Dict) -> str:
    """Create manipulation-specific prompt."""
    return f"""{MANIPULATION_SYSTEM_PROMPT}

Task: {task_description}

Object Properties:
{json.dumps(object_properties, indent=2)}

Robot State:
{json.dumps(robot_state, indent=2)}

Environment State:
{json.dumps(environment_state, indent=2)}

Create a detailed manipulation plan that accounts for the object properties, robot capabilities, and environmental constraints."""
```

### 4. Context Management and Memory

#### Conversation Memory for Context Preservation
```python
# context_management.py
from collections import deque
import time
from typing import Optional

class RoboticsConversationMemory:
    """
    Manages conversation context for multi-turn interactions with LLM.
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)
        self.context_snapshot = {}

    def add_interaction(self, user_input: str, robot_response: str,
                       action_taken: Optional[List[Dict]] = None):
        """Add an interaction to the conversation history."""
        interaction = {
            "timestamp": time.time(),
            "user_input": user_input,
            "robot_response": robot_response,
            "actions_taken": action_taken or [],
            "summary": self.summarize_interaction(user_input, robot_response, action_taken)
        }

        self.conversation_history.append(interaction)

    def get_recent_context(self, num_turns: int = 3) -> str:
        """Get recent conversation context for LLM prompts."""
        recent_interactions = list(self.conversation_history)[-num_turns:]

        context_parts = ["Recent Conversation History:"]
        for i, interaction in enumerate(recent_interactions):
            context_parts.append(f"Turn {len(self.conversation_history)-num_turns+i+1}:")
            context_parts.append(f"  User: {interaction['user_input']}")
            context_parts.append(f"  Robot: {interaction['robot_response']}")
            if interaction['actions_taken']:
                context_parts.append(f"  Actions: {[a['action'] for a in interaction['actions_taken']]}")
            context_parts.append("")

        return "\n".join(context_parts)

    def summarize_interaction(self, user_input: str, robot_response: str,
                            actions_taken: Optional[List]) -> str:
        """Create a summary of the interaction."""
        action_str = ", ".join([f"{a['action']}({a.get('parameters', {})})"
                               for a in actions_taken or []]) if actions_taken else "None"

        return f"User said '{user_input}', robot responded '{robot_response}', took actions: {action_str}"

    def update_context_snapshot(self, robot_state: Dict, environment_state: Dict):
        """Update the context snapshot with current states."""
        self.context_snapshot = {
            "last_updated": time.time(),
            "robot_state": robot_state,
            "environment_state": environment_state
        }

    def get_full_context_for_prompt(self) -> str:
        """Get full context for inclusion in LLM prompts."""
        context_parts = []

        # Add recent conversation history
        context_parts.append(self.get_recent_context())

        # Add current states if available
        if self.context_snapshot:
            context_parts.append("Current Robot State:")
            context_parts.append(json.dumps(self.context_snapshot.get("robot_state", {}), indent=2))
            context_parts.append("\nCurrent Environment State:")
            context_parts.append(json.dumps(self.context_snapshot.get("environment_state", {}), indent=2))

        return "\n".join(context_parts)

# Example usage in prompt engineering
class AdvancedRoboticsPromptEngineer(RoboticsPromptEngineer):
    """
    Advanced prompt engineer with conversation memory and context management.
    """

    def __init__(self):
        super().__init__()
        self.conversation_memory = RoboticsConversationMemory()

    def create_contextual_prompt(self, command: str, robot_state: Dict,
                               environment_state: Dict) -> str:
        """Create prompt with conversation history and context."""
        # Update context snapshot
        self.conversation_memory.update_context_snapshot(robot_state, environment_state)

        # Create base prompt
        base_prompt = self.create_task_decomposition_prompt(
            command, robot_state, environment_state
        )

        # Add conversation history
        context_section = self.conversation_memory.get_full_context_for_prompt()

        full_prompt = f"""{context_section}

{base_prompt}

Consider the conversation history and current context when generating the action sequence."""

        return full_prompt

    def process_successful_command(self, original_command: str,
                                 action_sequence: List[Dict]):
        """Process a successfully executed command to update memory."""
        response_text = f"Executed command: {original_command}"
        self.conversation_memory.add_interaction(
            original_command,
            response_text,
            action_sequence
        )

    def handle_command_failure(self, original_command: str,
                             error_message: str):
        """Handle command failure for learning purposes."""
        response_text = f"Failed to execute command: {error_message}"
        self.conversation_memory.add_interaction(
            original_command,
            response_text,
            None  # No actions taken due to failure
        )
```

### 5. Prompt Optimization Techniques

#### Dynamic Prompt Adjustment
```python
# prompt_optimizer.py
import re
from typing import Tuple

class PromptOptimizer:
    """
    Optimizes prompts based on success/failure patterns and performance metrics.
    """

    def __init__(self):
        self.success_patterns = []
        self.failure_patterns = []
        self.performance_metrics = {
            "avg_response_time": [],
            "success_rate": [],
            "token_usage": []
        }

    def analyze_response_quality(self, prompt: str, response: str,
                               execution_result: Dict) -> float:
        """
        Analyze the quality of an LLM response based on execution results.

        Returns a quality score between 0 and 1.
        """
        score = 0.0

        # Check if response is valid JSON
        try:
            parsed_response = json.loads(response)
            score += 0.3  # Valid structure
        except:
            return 0.0  # Invalid JSON is completely unusable

        # Check if actions are valid for robot
        if "action_sequence" in parsed_response:
            actions = parsed_response["action_sequence"]
            valid_actions = sum(1 for action in actions
                              if self.is_valid_robot_action(action.get("action", "")))
            score += 0.4 * (valid_actions / len(actions)) if actions else 0

        # Check execution success
        if execution_result.get("success", False):
            score += 0.3
        else:
            # Penalize for failure but don't eliminate completely
            score *= 0.7

        return min(score, 1.0)  # Cap at 1.0

    def is_valid_robot_action(self, action_name: str) -> bool:
        """Check if action name is valid for robot (simplified)."""
        # This would be connected to the actual robot capabilities
        valid_actions = {
            "navigate", "detect", "grasp", "transport", "place",
            "speak", "open_door", "close_door", "operate_switch"
        }
        return any(valid in action_name.lower() for valid in valid_actions)

    def optimize_prompt_for_domain(self, base_prompt: str,
                                  domain_examples: List[Tuple[str, str]]) -> str:
        """
        Optimize prompt based on domain-specific examples.

        Args:
            base_prompt: Original prompt template
            domain_examples: List of (input, desired_output) pairs

        Returns:
            Optimized prompt
        """
        # Analyze examples to identify patterns
        common_phrases = self.extract_common_phrases(domain_examples)
        frequent_mistakes = self.identify_frequent_mistakes(domain_examples)

        # Enhance prompt with domain-specific guidance
        enhanced_prompt = base_prompt

        if common_phrases:
            enhanced_prompt += f"\n\nCommon Command Patterns:\n{common_phrases}"

        if frequent_mistakes:
            enhanced_prompt += f"\n\nThings to Avoid:\n{frequent_mistakes}"

        return enhanced_prompt

    def extract_common_phrases(self, examples: List[Tuple[str, str]]) -> str:
        """Extract common phrases from successful examples."""
        # Simple heuristic: look for common verb patterns in commands
        command_phrases = []
        for input_text, _ in examples:
            # Extract common command patterns like "go to X", "pick up Y", etc.
            phrases = re.findall(r'(?:please|could you|can you|go to|pick up|bring me|move to|find|locate)\s+\w+', input_text.lower())
            command_phrases.extend(phrases)

        # Return most common phrases
        phrase_counts = {}
        for phrase in command_phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return "\n".join([f"- {phrase}" for phrase, count in sorted_phrases])

    def identify_frequent_mistakes(self, examples: List[Tuple[str, str]]) -> str:
        """Identify patterns that lead to mistakes."""
        # This would be more sophisticated in practice
        # For now, just return some common robotics-specific pitfalls
        return """- Using actions not available to the robot
- Ignoring environmental constraints
- Not considering robot's current state
- Proposing physically impossible maneuvers
- Forgetting to check for obstacles"""

# Integration with the main prompt engineer
class OptimizedRoboticsPromptEngineer(AdvancedRoboticsPromptEngineer):
    """
    Prompt engineer with optimization capabilities.
    """

    def __init__(self):
        super().__init__()
        self.optimizer = PromptOptimizer()
        self.domain_examples = []

    def add_domain_example(self, command: str, expected_action_sequence: List[Dict]):
        """Add a domain-specific example for optimization."""
        self.domain_examples.append((command, json.dumps(expected_action_sequence, indent=2)))

    def create_optimized_prompt(self, command: str, robot_state: Dict,
                              environment_state: Dict) -> str:
        """Create an optimized prompt based on domain knowledge."""
        # First, create the base contextual prompt
        base_prompt = self.create_contextual_prompt(
            command, robot_state, environment_state
        )

        # If we have domain examples, optimize the prompt
        if self.domain_examples:
            optimized_prompt = self.optimizer.optimize_prompt_for_domain(
                base_prompt,
                self.domain_examples
            )
            return optimized_prompt

        return base_prompt

    def record_execution_result(self, command: str, response: str,
                              execution_result: Dict):
        """Record execution result for future optimization."""
        quality_score = self.optimizer.analyze_response_quality(
            command, response, execution_result
        )

        print(f"Response quality score: {quality_score:.2f}")

        if quality_score < 0.5:
            print("Low-quality response detected, consider adding to training examples")
```

### 6. Integration with ROS 2

#### ROS 2 Node for LLM Integration
```python
#!/usr/bin/env python3
"""
ROS 2 node for LLM-based task decomposition.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from robotics_demo_msgs.msg import TaskSequence, RobotAction
import json
import time
from typing import Dict, Any

class LLMPromptNode(Node):
    """
    ROS 2 node that interfaces with LLM for natural language command processing.
    """

    def __init__(self):
        super().__init__('llm_prompt_node')

        # Parameters
        self.declare_parameter('llm_model', 'gpt-4')
        self.declare_parameter('max_tokens', 1000)
        self.declare_parameter('temperature', 0.3)
        self.declare_parameter('response_timeout', 30.0)

        self.llm_model = self.get_parameter('llm_model').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.temperature = self.get_parameter('temperature').value
        self.response_timeout = self.get_parameter('response_timeout').value

        # Publishers
        self.action_sequence_pub = self.create_publisher(
            TaskSequence, '/robot/action_sequence', 10
        )
        self.status_pub = self.create_publisher(
            String, '/llm/status', 10
        )

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/voice_commands', self.command_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Initialize prompt engineer
        self.prompt_engineer = OptimizedRoboticsPromptEngineer()

        # Robot state tracking
        self.robot_state = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            'battery_level': 1.0,
            'gripper_status': 'free',
            'current_task': 'idle'
        }

        self.environment_state = {
            'known_objects': {},
            'navigable_locations': [],
            'obstacles': []
        }

        self.get_logger().info('LLM Prompt Node initialized')

    def odom_callback(self, msg: Odometry):
        """Update robot position from odometry."""
        self.robot_state['position'] = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z
        }
        self.robot_state['orientation'] = {
            'x': msg.pose.pose.orientation.x,
            'y': msg.pose.pose.orientation.y,
            'z': msg.pose.pose.orientation.z,
            'w': msg.pose.pose.orientation.w
        }

    def joint_state_callback(self, msg: JointState):
        """Update robot state from joint states."""
        # Update gripper status based on joint positions
        for i, name in enumerate(msg.name):
            if 'gripper' in name.lower():
                if msg.position[i] > 0.5:  # Assuming >0.5 means open/gripper free
                    self.robot_state['gripper_status'] = 'free'
                else:
                    self.robot_state['gripper_status'] = 'occupied'

    def command_callback(self, msg: String):
        """Process incoming natural language command."""
        command = msg.data
        self.get_logger().info(f'Received command: "{command}"')

        # Publish status
        status_msg = String()
        status_msg.data = f'Processing command: {command}'
        self.status_pub.publish(status_msg)

        try:
            # Create contextual prompt
            prompt = self.prompt_engineer.create_optimized_prompt(
                command, self.robot_state, self.environment_state
            )

            # Call LLM (this would make an actual API call in real implementation)
            response = self.call_llm_api(prompt)

            # Parse response
            parsed_response = self.parse_llm_response(response)

            # Validate action sequence
            if self.prompt_engineer.validate_action_sequence(
                parsed_response["action_sequence"]
            ):
                # Publish action sequence
                task_seq_msg = self.create_task_sequence_message(
                    parsed_response["action_sequence"]
                )
                self.action_sequence_pub.publish(task_seq_msg)

                # Update conversation memory
                self.prompt_engineer.process_successful_command(
                    command, parsed_response["action_sequence"]
                )

                self.get_logger().info(
                    f'Published action sequence with {len(parsed_response["action_sequence"])} actions'
                )
            else:
                self.get_logger().error('Invalid action sequence generated by LLM')
                self.handle_invalid_response(command, parsed_response)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            self.handle_command_error(command, str(e))

    def call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with the given prompt.
        In a real implementation, this would make an actual API call.
        """
        # For this example, we'll return a mock response
        # In real implementation, you would use:
        # - OpenAI API: openai.ChatCompletion.create()
        # - Anthropic API: anthropic.Client().messages.create()
        # - Or other LLM provider API

        self.get_logger().info('Making LLM API call...')

        # Simulate API call delay
        time.sleep(1.0)

        # Mock response - in real implementation, replace with actual API call
        return self.get_mock_response()

    def get_mock_response(self) -> str:
        """Mock LLM response for demonstration purposes."""
        return '''{
    "reasoning": "The user wants the robot to navigate to the kitchen, detect a cup, pick it up, and bring it to the living room to place it on the table.",
    "action_sequence": [
        {
            "step": 1,
            "action": "navigation_to_position",
            "parameters": {"location": "kitchen"},
            "reason": "Navigate to kitchen to find cup"
        },
        {
            "step": 2,
            "action": "object_detection",
            "parameters": {"object_type": "cup"},
            "reason": "Detect cup in kitchen"
        },
        {
            "step": 3,
            "action": "object_grasping",
            "parameters": {"object": "cup", "position": "handle"},
            "reason": "Grasp the cup"
        },
        {
            "step": 4,
            "action": "navigation_to_position",
            "parameters": {"location": "living_room"},
            "reason": "Navigate to living room"
        },
        {
            "step": 5,
            "action": "object_placement",
            "parameters": {"object": "cup", "location": "table"},
            "reason": "Place cup on table"
        }
    ],
    "estimated_completion_time": 90,
    "potential_challenges": ["Cup might be in cabinet", "Table might be occupied"]
}'''

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured format."""
        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse LLM response as JSON: {e}')
            self.get_logger().debug(f'Response content: {response}')

            # Return a default structure in case of parsing failure
            return {
                "reasoning": "Failed to parse LLM response",
                "action_sequence": [],
                "estimated_completion_time": 0,
                "potential_challenges": ["Response parsing failed"]
            }

    def create_task_sequence_message(self, action_sequence: List[Dict]) -> TaskSequence:
        """Convert action sequence to ROS message."""
        task_seq_msg = TaskSequence()
        task_seq_msg.header.stamp = self.get_clock().now().to_msg()
        task_seq_msg.header.frame_id = "map"

        for action_dict in action_sequence:
            action_msg = RobotAction()
            action_msg.action_name = action_dict.get("action", "")
            action_msg.reason = action_dict.get("reason", "")

            # Convert parameters to string format
            params = action_dict.get("parameters", {})
            action_msg.parameters_json = json.dumps(params)

            task_seq_msg.actions.append(action_msg)

        return task_seq_msg

    def handle_invalid_response(self, command: str, response: Dict):
        """Handle invalid response from LLM."""
        status_msg = String()
        status_msg.data = f'Invalid response for command: {command}'
        self.status_pub.publish(status_msg)

        # Add to conversation memory as failure
        self.prompt_engineer.handle_command_failure(command, "Invalid action sequence")

    def handle_command_error(self, command: str, error: str):
        """Handle error during command processing."""
        status_msg = String()
        status_msg.data = f'Error processing command: {error}'
        self.status_pub.publish(status_msg)

        # Add to conversation memory as failure
        self.prompt_engineer.handle_command_failure(command, error)


def main(args=None):
    rclpy.init(args=args)

    node = LLMPromptNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM Prompt Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercises

1. Implement the basic prompt engineering framework with context management
2. Create specialized prompts for different robot functions (navigation, manipulation, etc.)
3. Develop a conversation memory system for multi-turn interactions
4. Implement prompt optimization based on execution feedback
5. Integrate the LLM prompt system with ROS 2 messaging
6. Test the system with various natural language commands
7. Evaluate the effectiveness of different prompt engineering techniques
8. Optimize prompt length and structure for real-time performance

## References

1. OpenAI Prompt Engineering Guide: https://platform.openai.com/docs/guides/prompt-engineering
2. Chain-of-Thought Prompting: https://arxiv.org/abs/2201.11903
3. ReAct: Synergizing Reasoning and Acting: https://arxiv.org/abs/2210.03629
4. ROS 2 Documentation: https://docs.ros.org/en/humble/

## Further Reading

- Advanced reasoning techniques for robotics applications
- Multi-modal prompting combining vision and language
- Safety considerations in LLM-robot integration
- Real-time optimization of prompt engineering strategies
- Evaluation metrics for LLM-based robot task planning
- Integration with symbolic planning systems