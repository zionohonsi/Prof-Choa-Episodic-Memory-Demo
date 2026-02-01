"""
Example usage of the Cognitive Architecture system.
Demonstrates how to use the system for problem-solving.
"""

from cognitive_system import CognitiveSystem, MockLLM
import json


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)
    print()


def demonstrate_basic_usage():
    """Demonstrate basic usage of the cognitive system."""
    print_separator("BASIC USAGE DEMONSTRATION")
    
    # Initialize the system
    print("Initializing cognitive system...")
    system = CognitiveSystem()
    
    # Add some knowledge to semantic memory
    system.add_knowledge("Machine learning models can be trained using supervised, unsupervised, or reinforcement learning.")
    system.add_knowledge("Neural networks consist of layers of interconnected nodes that process information.")
    system.add_knowledge("Meta-learning enables models to learn how to learn from limited data.")
    
    # Solve a problem
    goal = "Design a novel machine learning architecture that can learn from very few examples"
    print(f"Goal: {goal}\n")
    
    print("Running cognitive loop...")
    episode = system.solve(goal, max_steps=10)
    
    # Display results
    print_separator("RESULTS")
    print(f"Outcome: {episode.outcome}")
    print(f"\nGoal: {episode.goal}")
    print(f"\nSteps taken: {len(episode.steps)}")
    
    print("\nStep-by-step trace:")
    for i, (action, result) in enumerate(episode.steps, 1):
        print(f"\n  Step {i}: {action}")
        if isinstance(result, list):
            for item in result[:3]:  # Show first 3 items
                print(f"    - {str(item)[:100]}...")
        else:
            result_str = str(result)[:200]
            print(f"    Result: {result_str}...")
    
    print("\n\nFinal working memory state:")
    wm_state = system.get_working_memory_state()
    print(f"  Goal: {wm_state['goal']}")
    print(f"  Subgoals: {len(wm_state['subgoals'])}")
    for i, sg in enumerate(wm_state['subgoals'], 1):
        print(f"    {i}. {sg}")
    
    print(f"\n  Hypotheses: {len(wm_state['hypotheses'])}")
    for i, h in enumerate(wm_state['hypotheses'], 1):
        print(f"    {i}. {h[:100]}...")
    
    print(f"\n  Context notes: {len(wm_state['context_notes'])}")
    for note in wm_state['context_notes']:
        print(f"    - {note}")


def demonstrate_episodic_learning():
    """Demonstrate how the system learns from episodes."""
    print_separator("EPISODIC LEARNING DEMONSTRATION")
    
    system = CognitiveSystem()
    
    # Solve multiple related problems
    problems = [
        "Invent a new optimization algorithm for deep learning",
        "Design a neural network architecture for few-shot learning",
        "Create a novel approach to transfer learning"
    ]
    
    print("Solving multiple problems to build episodic memory...\n")
    
    for i, problem in enumerate(problems, 1):
        print(f"Problem {i}: {problem}")
        episode = system.solve(problem, max_steps=8)
        print(f"  Outcome: {episode.outcome}")
        print(f"  Steps: {len(episode.steps)}\n")
    
    # Check episodic memory
    episodes = system.get_episodes()
    print(f"Total episodes stored: {len(episodes)}")
    
    print("\nEpisode summaries:")
    for i, ep in enumerate(episodes, 1):
        print(f"\n  Episode {i}:")
        print(f"    Goal: {ep.goal}")
        print(f"    Outcome: {ep.outcome}")
        print(f"    Steps: {len(ep.steps)}")
        print(f"    Notes: {', '.join(ep.notes) if ep.notes else 'None'}")


def demonstrate_with_real_llm():
    """
    Demonstrate how to integrate with a real LLM (OpenAI).
    This requires OPENAI_API_KEY environment variable to be set.
    """
    print_separator("REAL LLM INTEGRATION (Optional)")
    
    try:
        from openai import OpenAI
        import os
        
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not found. Skipping real LLM demonstration.")
            print("To use real LLM, set OPENAI_API_KEY environment variable.")
            return
        
        class OpenAILLM:
            """Wrapper for OpenAI API."""
            
            def __init__(self, model="gpt-4.1-mini"):
                self.client = OpenAI()
                self.model = model
            
            def generate(self, system_prompt: str, user_prompt: str) -> str:
                """Generate response using OpenAI API."""
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
        
        print("Initializing system with OpenAI GPT-4...")
        llm = OpenAILLM()
        system = CognitiveSystem(llm=llm)
        
        goal = "Invent a new data structure optimized for graph neural networks"
        print(f"\nGoal: {goal}\n")
        
        print("Running with real LLM (this may take a minute)...")
        episode = system.solve(goal, max_steps=8)
        
        print_separator("REAL LLM RESULTS")
        print(f"Outcome: {episode.outcome}")
        print(f"\nFinal hypotheses:")
        wm_state = system.get_working_memory_state()
        for i, h in enumerate(wm_state['hypotheses'], 1):
            print(f"\n{i}. {h}")
        
    except ImportError:
        print("OpenAI library not available. Install with: pip install openai")
    except Exception as e:
        print(f"Error with real LLM: {e}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" COGNITIVE ARCHITECTURE DEMONSTRATION")
    print("="*70)
    
    # Basic usage
    demonstrate_basic_usage()
    
    # Episodic learning
    demonstrate_episodic_learning()
    
    # Real LLM (optional)
    demonstrate_with_real_llm()
    
    print_separator("DEMONSTRATION COMPLETE")
    print("The cognitive system successfully:")
    print("  ✓ Decomposed goals into subgoals")
    print("  ✓ Generated multiple hypotheses")
    print("  ✓ Evaluated and refined ideas")
    print("  ✓ Stored episodes for future learning")
    print("  ✓ Retrieved relevant knowledge and past experiences")
    print("\nYou can now customize this system for your specific domain!")


if __name__ == "__main__":
    main()
