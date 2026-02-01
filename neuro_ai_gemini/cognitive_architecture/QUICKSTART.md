# Quick Start Guide

Get started with the Cognitive Architecture system in 5 minutes.

## Installation

```bash
# No dependencies required for basic usage with MockLLM
cd cognitive_architecture

# Optional: Install OpenAI for real LLM integration
pip install openai
```

## Basic Usage

### 1. Simple Problem Solving

```python
from cognitive_system import CognitiveSystem

# Create the system
system = CognitiveSystem()

# Solve a problem
goal = "Design a new machine learning algorithm"
episode = system.solve(goal, max_steps=10)

# Check results
print(f"Outcome: {episode.outcome}")
print(f"Steps: {len(episode.steps)}")
```

### 2. Adding Domain Knowledge

```python
# Add knowledge to semantic memory
system.add_knowledge("Neural networks use gradient descent")
system.add_knowledge("Transformers use attention mechanisms")

# Solve with knowledge context
episode = system.solve("Improve transformer efficiency")
```

### 3. Learning from Multiple Episodes

```python
# Solve related problems
problems = [
    "Optimize neural network training",
    "Design efficient attention mechanism",
    "Create fast inference pipeline"
]

for problem in problems:
    episode = system.solve(problem)
    print(f"{problem}: {episode.outcome}")

# View all episodes
episodes = system.get_episodes()
print(f"Total episodes: {len(episodes)}")
```

### 4. Using Real LLMs

```python
from cognitive_system import CognitiveSystem
from openai import OpenAI

class OpenAILLM:
    def __init__(self):
        self.client = OpenAI()
    
    def generate(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

# Use with real LLM
system = CognitiveSystem(llm=OpenAILLM())
episode = system.solve("Your goal here")
```

## Running Examples

### Basic Example
```bash
python example_usage.py
```

### Advanced Examples
```bash
python advanced_example.py
```

### Unit Tests
```bash
python test_cognitive_system.py
```

## Understanding the Output

When you run `system.solve(goal)`, the system:

1. **Decomposes** the goal into subgoals
2. **Generates** multiple hypothesis solutions
3. **Evaluates** the hypotheses
4. **Refines** the best hypothesis
5. **Stores** the episode for future learning

### Episode Structure

```python
episode = system.solve("Your goal")

# Access episode data
print(episode.goal)           # The original goal
print(episode.outcome)        # "success", "partial", or "failure"
print(episode.steps)          # List of (action, result) tuples
print(episode.notes)          # Context notes from the process
```

### Working Memory State

```python
state = system.get_working_memory_state()

print(state['goal'])          # Current goal
print(state['subgoals'])      # List of subgoals
print(state['hypotheses'])    # Generated hypotheses
print(state['context_notes']) # Accumulated notes
```

## Customization

### Custom Actions

```python
from cognitive_system import Controller

class CustomController(Controller):
    def execute_action(self, action):
        if action == "MY_CUSTOM_ACTION":
            # Your custom logic
            return self.skill_core.call("custom_role", "custom_prompt")
        return super().execute_action(action)
```

### Domain-Specific System

```python
# Create specialized system
class MyDomainSystem(CognitiveSystem):
    def __init__(self):
        super().__init__()
        # Add domain knowledge
        self.add_knowledge("Domain fact 1")
        self.add_knowledge("Domain fact 2")
    
    def solve_domain_problem(self, problem):
        # Custom preprocessing
        enhanced_goal = f"In the context of [domain]: {problem}"
        return self.solve(enhanced_goal)
```

## Common Patterns

### Pattern 1: Iterative Refinement

```python
system = CognitiveSystem()

# First attempt
episode1 = system.solve("Design algorithm", max_steps=5)

# Refine based on first attempt
episode2 = system.solve(
    f"Improve this design: {episode1.steps[-1][1]}", 
    max_steps=5
)
```

### Pattern 2: Batch Processing

```python
goals = ["Goal 1", "Goal 2", "Goal 3"]
results = []

for goal in goals:
    episode = system.solve(goal, max_steps=8)
    results.append({
        'goal': goal,
        'outcome': episode.outcome,
        'best_hypothesis': system.get_working_memory_state()['hypotheses'][0]
    })
```

### Pattern 3: Knowledge Accumulation

```python
system = CognitiveSystem()

# Solve and learn
for problem in problem_set:
    episode = system.solve(problem)
    
    # Extract learnings
    if episode.outcome == "success":
        best_solution = system.get_working_memory_state()['hypotheses'][0]
        system.add_knowledge(f"For {problem}: {best_solution}")
```

## Tips

1. **Start with MockLLM** to understand the flow before using real LLMs
2. **Adjust max_steps** based on problem complexity (5-20 typical)
3. **Add domain knowledge** before solving for better results
4. **Check working memory** to see intermediate reasoning
5. **Store episodes** to enable learning over time

## Troubleshooting

### Issue: No hypotheses generated
**Solution**: Increase `max_steps` or check if goal decomposition succeeded

### Issue: Poor quality outputs with MockLLM
**Solution**: MockLLM returns fixed responses; use real LLM for actual problems

### Issue: System gets stuck
**Solution**: Implement custom `stuck()` method in Controller to detect and break loops

## Next Steps

- Read [README.md](README.md) for full documentation
- Explore [advanced_example.py](advanced_example.py) for customization ideas
- Integrate with your preferred LLM provider
- Add domain-specific actions and knowledge
- Implement learned controller policy with RL

## Support

For issues or questions:
- Check the README.md for detailed documentation
- Review example files for usage patterns
- Examine test files for API reference
