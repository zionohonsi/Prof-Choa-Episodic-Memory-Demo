# Cognitive Architecture: Working Memory System with LLM-based Skill Engine

A Python implementation of a cognitive architecture that enables learning, simulation, and invention through a working memory system powered by large language models (LLMs).

## Overview

This implementation is based on the cognitive architecture prototype that treats an LLM as a **skill engine** surrounded by memory systems and a controller that orchestrates problem-solving. The system can:

- **Learn** from past experiences through episodic memory
- **Simulate** and evaluate hypotheses iteratively
- **Invent** new solutions by generating, evaluating, and refining ideas
- **Adapt** over time as it accumulates more episodes

## Architecture

```
        Controller
            ↕
WorkingMemory ↔ SkillCore (LLM)
            ↕
EpisodicMemory, SemanticMemory
```

### Core Components

1. **SkillCore (LLM)**: The procedural engine that generates plans, hypotheses, evaluations, and refinements
2. **WorkingMemory**: Dynamic scratchpad holding current goal, subgoals, hypotheses, and intermediate results
3. **EpisodicMemory**: Stores past problem-solving episodes for retrieval and learning
4. **SemanticMemory**: Stores factual knowledge and domain information
5. **Controller**: Orchestrates the system by selecting actions and managing the cognitive loop

## Key Features

### Action Space

The controller can select from these actions:

- **DECOMPOSE_GOAL**: Break down the main goal into concrete subgoals
- **RETRIEVE_EPISODES**: Pull similar past problem-solving attempts
- **RETRIEVE_KNOWLEDGE**: Query semantic memory for relevant facts
- **GENERATE_HYPOTHESES**: Create candidate solutions or ideas
- **EVALUATE_HYPOTHESES**: Critique and rank the generated hypotheses
- **REFINE_TOP_HYPOTHESIS**: Iteratively improve the best hypothesis
- **SUMMARIZE_PROGRESS**: Compress current state into context notes

### Cognitive Loop

The system operates in an iterative loop:

1. **Select** an action based on current working memory state
2. **Execute** the action using the LLM or memory systems
3. **Update** working memory with the results
4. **Repeat** until goal is satisfied or max steps reached
5. **Store** the episode for future learning

## Installation

### Requirements

- Python 3.7+
- (Optional) OpenAI API key for using real LLMs

### Setup

```bash
# Clone or download the files
cd cognitive_architecture

# (Optional) Install OpenAI for real LLM integration
pip install openai
```

## Usage

### Basic Usage

```python
from cognitive_system import CognitiveSystem

# Initialize the system
system = CognitiveSystem()

# Add domain knowledge
system.add_knowledge("Neural networks learn through backpropagation")
system.add_knowledge("Meta-learning enables learning from few examples")

# Solve a problem
goal = "Design a novel machine learning architecture for few-shot learning"
episode = system.solve(goal, max_steps=10)

# Check results
print(f"Outcome: {episode.outcome}")
print(f"Steps taken: {len(episode.steps)}")

# View working memory state
wm_state = system.get_working_memory_state()
print(f"Hypotheses generated: {wm_state['hypotheses']}")
```

### Using Real LLMs (OpenAI)

```python
from cognitive_system import CognitiveSystem
from openai import OpenAI

class OpenAILLM:
    def __init__(self, model="gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
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

# Initialize with real LLM
llm = OpenAILLM()
system = CognitiveSystem(llm=llm)

# Solve problems with real AI
episode = system.solve("Invent a new optimization algorithm", max_steps=15)
```

### Running Examples

```bash
# Run the demonstration script
python example_usage.py
```

This will demonstrate:
- Basic problem-solving workflow
- Episodic memory accumulation
- (Optional) Integration with real LLMs

## Extending the System

### 1. Custom LLM Integration

Implement any LLM by creating a class with a `generate` method:

```python
class CustomLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Your LLM implementation
        return response_text

system = CognitiveSystem(llm=CustomLLM())
```

### 2. Learning Controller Policy

Replace the heuristic `select_action` method with a learned policy:

```python
class LearnedController(Controller):
    def __init__(self, *args, policy_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_model = policy_model
    
    def select_action(self) -> str:
        state_features = self._extract_state_features()
        action = self.policy_model.predict(state_features)
        return action
    
    def _extract_state_features(self):
        # Extract features from working memory
        return {
            'num_subgoals': len(self.working_mem.subgoals),
            'num_hypotheses': len(self.working_mem.hypotheses),
            'num_evaluations': len(self.working_mem.intermediate_results),
            'step_count': len(self.trace)
        }
```

### 3. Domain-Specific Actions

Add custom actions for your domain:

```python
# Add to ACTIONS list
ACTIONS.append("RUN_SIMULATION")
ACTIONS.append("GENERATE_CODE")

# Implement in execute_action
def execute_action(self, action: str) -> Any:
    # ... existing actions ...
    
    if action == "RUN_SIMULATION":
        # Generate simulation code
        code = self.skill_core.call("code_generator", prompt)
        # Execute and return results
        return run_code(code)
    
    elif action == "GENERATE_CODE":
        # Generate implementation code
        return self.skill_core.call("programmer", prompt)
```

### 4. Enhanced Memory Systems

Improve memory retrieval with embeddings:

```python
from sentence_transformers import SentenceTransformer

class EmbeddingEpisodicMemory(EpisodicMemory):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = []
    
    def store_episode(self, episode: Episode):
        super().store_episode(episode)
        embedding = self.encoder.encode(episode.goal)
        self.embeddings.append(embedding)
    
    def retrieve_similar(self, goal: str, k: int = 3):
        query_embedding = self.encoder.encode(goal)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [self.episodes[i] for i in top_k_indices]
```

## How It Learns

### Current Implementation (Heuristic)

The controller uses simple heuristics to select actions based on working memory state.

### Future Learning Mechanisms

1. **Reinforcement Learning**: Train a policy network to select actions based on episode outcomes
   - State: Working memory features
   - Actions: Available cognitive actions
   - Reward: Based on episode success and efficiency

2. **Episode Mining**: Extract patterns from successful episodes
   - Identify common action sequences
   - Build meta-strategies for problem types
   - Use as few-shot examples in prompts

3. **Meta-Learning**: Learn how to learn across problem domains
   - Adapt action selection strategies
   - Transfer knowledge between domains
   - Improve hypothesis generation quality

## Applications

This architecture can be applied to:

- **Algorithm Discovery**: Invent new algorithms or optimization techniques
- **Architecture Design**: Create novel neural network architectures
- **Scientific Hypothesis Generation**: Generate and test scientific hypotheses
- **Creative Problem Solving**: Tackle open-ended creative challenges
- **Code Generation**: Design and implement software solutions
- **Game Design**: Invent new game mechanics or rules
- **Material Discovery**: Propose new materials with desired properties

## Limitations and Future Work

### Current Limitations

- Heuristic action selection (not learned)
- Simple keyword-based memory retrieval
- No external tool integration for simulation/validation
- Limited context management for long episodes

### Planned Enhancements

- [ ] Learned controller policy using RL
- [ ] Embedding-based semantic search
- [ ] External tool integration (code execution, simulation)
- [ ] Multi-modal memory (images, diagrams, code)
- [ ] Hierarchical goal decomposition
- [ ] Collaborative multi-agent problem solving
- [ ] Real-time learning from human feedback

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                      Controller                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Action Selection (Gating)                         │  │
│  │ - Heuristic or Learned Policy                     │  │
│  └───────────────────────────────────────────────────┘  │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐    ┌──────────────────────────┐
│   Working Memory       │◄──►│   SkillCore (LLM)        │
│  ┌──────────────────┐  │    │  ┌────────────────────┐  │
│  │ Goal             │  │    │  │ Planner            │  │
│  │ Subgoals         │  │    │  │ Inventor           │  │
│  │ Hypotheses       │  │    │  │ Critic             │  │
│  │ Results          │  │    │  │ Refiner            │  │
│  │ Context Notes    │  │    │  │ Summarizer         │  │
│  └──────────────────┘  │    │  └────────────────────┘  │
└────────────┬───────────┘    └──────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│                  Memory Systems                        │
│  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │ Episodic Memory      │  │ Semantic Memory      │   │
│  │ - Past episodes      │  │ - Facts              │   │
│  │ - Solutions          │  │ - Knowledge base     │   │
│  │ - Outcomes           │  │ - Domain info        │   │
│  └──────────────────────┘  └──────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

## Citation

This implementation is based on the cognitive architecture prototype described in the provided document. If you use this code in your research, please cite appropriately.

## License

This implementation is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:

- Better parsing of LLM outputs
- More sophisticated action selection strategies
- Integration with specific LLM providers
- Domain-specific customizations
- Performance optimizations
- Additional memory systems

## Contact

For questions or suggestions about this implementation, please open an issue or submit a pull request.
