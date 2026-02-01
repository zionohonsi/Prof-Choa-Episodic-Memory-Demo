"""
Advanced Example: Domain-Specific Customization
Demonstrates how to extend the cognitive system for specific domains.
"""

from cognitive_system import (
    CognitiveSystem, Controller, SkillCore, WorkingMemory,
    EpisodicMemory, SemanticMemory, MockLLM, ACTIONS
)
from typing import Any, List, Dict
import json


# ============================================================================
# Example 1: Neural Architecture Search Domain
# ============================================================================

class NeuralArchitectureSearchController(Controller):
    """
    Specialized controller for neural architecture search.
    Adds domain-specific actions like GENERATE_ARCHITECTURE and SIMULATE_TRAINING.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add domain-specific actions
        self.domain_actions = [
            "GENERATE_ARCHITECTURE",
            "SIMULATE_TRAINING",
            "ANALYZE_COMPLEXITY"
        ]
    
    def select_action(self) -> str:
        """Enhanced action selection with domain-specific logic."""
        wm = self.working_mem
        
        # Domain-specific heuristics
        if not wm.subgoals:
            return "DECOMPOSE_GOAL"
        
        if "architecture" in wm.goal.lower() and not wm.hypotheses:
            return "GENERATE_ARCHITECTURE"
        
        if wm.hypotheses and "simulate" not in str(self.trace):
            return "SIMULATE_TRAINING"
        
        if len(wm.intermediate_results) > 0 and "analyze" not in str(self.trace):
            return "ANALYZE_COMPLEXITY"
        
        # Fall back to base logic
        return super().select_action()
    
    def execute_action(self, action: str) -> Any:
        """Execute domain-specific actions."""
        wm = self.working_mem
        
        if action == "GENERATE_ARCHITECTURE":
            prompt = f"""Design 3 novel neural network architectures for: {wm.goal}

Consider:
- Layer types (conv, attention, MLP, etc.)
- Skip connections and residual paths
- Parameter efficiency
- Computational complexity

Provide concrete architecture specifications."""
            out = self.skill_core.call("architect", prompt)
            return self._parse_hypotheses(out)
        
        elif action == "SIMULATE_TRAINING":
            if not wm.hypotheses:
                return "No architectures to simulate"
            
            # Simulate training performance (mock)
            results = []
            for i, arch in enumerate(wm.hypotheses[:3]):
                mock_accuracy = 85 + i * 3  # Mock results
                mock_params = 10 - i * 2
                results.append({
                    "architecture": arch[:50] + "...",
                    "accuracy": mock_accuracy,
                    "params_millions": mock_params,
                    "training_time_hours": 5 + i
                })
            return results
        
        elif action == "ANALYZE_COMPLEXITY":
            prompt = f"""Analyze the computational complexity and efficiency of these architectures:

{json.dumps(wm.intermediate_results[-1] if wm.intermediate_results else {}, indent=2)}

Provide analysis of:
- Time complexity
- Space complexity
- Scalability
- Trade-offs"""
            out = self.skill_core.call("analyzer", prompt)
            return out
        
        # Fall back to base actions
        return super().execute_action(action)


def demonstrate_neural_architecture_search():
    """Demonstrate neural architecture search customization."""
    print("\n" + "="*70)
    print(" NEURAL ARCHITECTURE SEARCH EXAMPLE")
    print("="*70 + "\n")
    
    # Create system with custom controller
    llm = MockLLM()
    skill_core = SkillCore(llm)
    working_mem = WorkingMemory()
    episodic_mem = EpisodicMemory()
    semantic_mem = SemanticMemory([
        "Transformers use self-attention mechanisms",
        "ResNet introduced skip connections",
        "EfficientNet optimizes for parameter efficiency"
    ])
    
    controller = NeuralArchitectureSearchController(
        skill_core, working_mem, episodic_mem, semantic_mem
    )
    
    # Solve architecture design problem
    controller.set_goal("Design an efficient neural architecture for image classification on mobile devices")
    episode = controller.run_until_done(max_steps=8)
    
    print(f"Outcome: {episode.outcome}")
    print(f"\nSteps taken:")
    for i, (action, result) in enumerate(episode.steps, 1):
        print(f"  {i}. {action}")
        if action == "SIMULATE_TRAINING" and isinstance(result, list):
            print("     Training simulation results:")
            for res in result:
                print(f"       - Accuracy: {res['accuracy']}%, Params: {res['params_millions']}M")
    
    print(f"\nFinal architectures generated: {len(working_mem.hypotheses)}")


# ============================================================================
# Example 2: Learning from Episode Patterns
# ============================================================================

class PatternLearningSystem(CognitiveSystem):
    """
    System that learns patterns from past episodes to improve future performance.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_patterns = {}  # Store successful action sequences
    
    def solve(self, goal: str, max_steps: int = 20):
        """Solve with pattern learning."""
        episode = super().solve(goal, max_steps)
        
        # Learn from successful episodes
        if episode.outcome == "success":
            self._extract_pattern(episode)
        
        return episode
    
    def _extract_pattern(self, episode):
        """Extract action sequence patterns from successful episodes."""
        # Extract action sequence
        action_sequence = [action for action, _ in episode.steps]
        
        # Store pattern by goal type
        goal_type = self._classify_goal(episode.goal)
        if goal_type not in self.action_patterns:
            self.action_patterns[goal_type] = []
        
        self.action_patterns[goal_type].append(action_sequence)
    
    def _classify_goal(self, goal: str) -> str:
        """Classify goal into categories."""
        goal_lower = goal.lower()
        if "design" in goal_lower or "architecture" in goal_lower:
            return "design"
        elif "optimize" in goal_lower or "improve" in goal_lower:
            return "optimization"
        elif "invent" in goal_lower or "create" in goal_lower:
            return "invention"
        else:
            return "general"
    
    def get_patterns(self) -> Dict[str, List[List[str]]]:
        """Get learned patterns."""
        return self.action_patterns
    
    def suggest_actions(self, goal: str) -> List[str]:
        """Suggest action sequence based on learned patterns."""
        goal_type = self._classify_goal(goal)
        if goal_type in self.action_patterns:
            # Return most common pattern
            patterns = self.action_patterns[goal_type]
            if patterns:
                return patterns[-1]  # Return most recent pattern
        return []


def demonstrate_pattern_learning():
    """Demonstrate learning from episode patterns."""
    print("\n" + "="*70)
    print(" PATTERN LEARNING EXAMPLE")
    print("="*70 + "\n")
    
    system = PatternLearningSystem()
    
    # Solve multiple design problems
    design_problems = [
        "Design a new data structure for graph processing",
        "Design an efficient caching algorithm",
        "Design a novel neural network layer"
    ]
    
    print("Solving design problems to learn patterns...\n")
    for problem in design_problems:
        episode = system.solve(problem, max_steps=6)
        print(f"✓ {problem}")
        print(f"  Outcome: {episode.outcome}, Steps: {len(episode.steps)}\n")
    
    # Check learned patterns
    patterns = system.get_patterns()
    print("Learned patterns:")
    for goal_type, pattern_list in patterns.items():
        print(f"\n  {goal_type.upper()} problems:")
        for i, pattern in enumerate(pattern_list, 1):
            print(f"    Pattern {i}: {' → '.join(pattern)}")
    
    # Suggest actions for new problem
    new_problem = "Design a distributed consensus algorithm"
    suggested = system.suggest_actions(new_problem)
    print(f"\n\nFor new problem: '{new_problem}'")
    print(f"Suggested action sequence: {' → '.join(suggested)}")


# ============================================================================
# Example 3: Multi-Step Simulation and Validation
# ============================================================================

class SimulationController(Controller):
    """Controller that can run simulations and validate hypotheses."""
    
    def execute_action(self, action: str) -> Any:
        """Add simulation capabilities."""
        if action == "REFINE_TOP_HYPOTHESIS":
            # After refinement, validate with simulation
            result = super().execute_action(action)
            
            # Run validation
            validation_prompt = f"""Validate this hypothesis with a thought experiment:

Hypothesis: {result}

Consider:
- Edge cases
- Potential failures
- Scalability issues
- Practical constraints

Provide validation results."""
            
            validation = self.skill_core.call("validator", validation_prompt)
            return f"{result}\n\nValidation: {validation}"
        
        return super().execute_action(action)


def demonstrate_simulation_validation():
    """Demonstrate simulation and validation."""
    print("\n" + "="*70)
    print(" SIMULATION & VALIDATION EXAMPLE")
    print("="*70 + "\n")
    
    llm = MockLLM()
    skill_core = SkillCore(llm)
    working_mem = WorkingMemory()
    episodic_mem = EpisodicMemory()
    semantic_mem = SemanticMemory()
    
    controller = SimulationController(
        skill_core, working_mem, episodic_mem, semantic_mem
    )
    
    controller.set_goal("Invent a new sorting algorithm optimized for nearly-sorted data")
    episode = controller.run_until_done(max_steps=8)
    
    print(f"Outcome: {episode.outcome}")
    print(f"\nFinal refined and validated hypothesis:")
    if working_mem.hypotheses:
        print(f"\n{working_mem.hypotheses[0]}")


# ============================================================================
# Main Demonstration
# ============================================================================

def main():
    """Run all advanced demonstrations."""
    print("\n" + "="*70)
    print(" ADVANCED COGNITIVE ARCHITECTURE EXAMPLES")
    print("="*70)
    
    # Example 1: Domain-specific customization
    demonstrate_neural_architecture_search()
    
    # Example 2: Pattern learning
    demonstrate_pattern_learning()
    
    # Example 3: Simulation and validation
    demonstrate_simulation_validation()
    
    print("\n" + "="*70)
    print(" ADVANCED DEMONSTRATIONS COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("  ✓ Controllers can be extended with domain-specific actions")
    print("  ✓ Systems can learn patterns from successful episodes")
    print("  ✓ Hypotheses can be validated through simulation")
    print("  ✓ The architecture is highly customizable for any domain")
    print("\nNext steps:")
    print("  • Integrate real LLMs for better hypothesis generation")
    print("  • Add reinforcement learning for controller policy")
    print("  • Implement actual simulation environments")
    print("  • Connect to external tools and APIs")


if __name__ == "__main__":
    main()
