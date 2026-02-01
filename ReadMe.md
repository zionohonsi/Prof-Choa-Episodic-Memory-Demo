# Neuro-AI Research System Guide

## Overview

This specialized cognitive architecture system is designed to solve problems in **AI, neuroscience, medicine, and brain-related topics**. It uses real LLMs (OpenAI GPT-4) to generate research insights and builds an **episodic memory library** that improves with each problem solved.

## Key Features

### ✅ Real LLM Integration
- Uses Google Gemini (gemini-2.0-flash-exp) to generate high-quality research insights
- Can pull from vast knowledge of AI, neuroscience, and medical literature
- Generates creative hypotheses and cross-domain connections
- Free tier available with generous quota

### ✅ Episodic Memory Building
- Stores every problem-solving episode
- Retrieves similar past experiences when solving new problems
- Gets better over time as memory grows

### ✅ Domain-Specific Actions
Beyond the standard cognitive actions, includes:
- **LITERATURE_SEARCH** - Identifies key concepts and recent research
- **ANALYZE_MECHANISMS** - Deep dive into biological/computational mechanisms
- **PROPOSE_EXPERIMENTS** - Suggests validation experiments
- **CROSS_DOMAIN_SYNTHESIS** - Connects AI, neuroscience, and medicine

### ✅ Progressive Problem Solving
- Start with foundational problems to build memory
- Tackle increasingly advanced problems
- System leverages past experiences for better solutions

## Setup

### 1. Prerequisites

```bash
# Make sure you have the cognitive architecture
cd /home/ubuntu/cognitive_architecture

# Install Google Generative AI (required for Gemini)
pip install google-generativeai
# OR use virtual environment:
python3 -m venv venv
source venv/bin/activate
pip install google-generativeai
```

### 2. Set Gemini API Key

```bash
# Set your API key
export GEMINI_API_KEY='your-api-key-here'

# Or add to your ~/.bashrc for persistence
echo "export GEMINI_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

**Get your Gemini API key:** https://aistudio.google.com/app/apikey

### 3. Verify Setup

```bash
python3 -c "import os; print('✓ API key set' if os.getenv('GEMINI_API_KEY') else '✗ API key not found')"
```

## Usage

### Option 1: Automated Demo (Recommended First Run)

Runs a complete demonstration that builds episodic memory and solves advanced problems:

```bash
python3 neuro_ai_research_system.py
```

**What it does:**
1. Builds episodic memory with 5 foundational problems
2. Solves 3 advanced problems using that memory
3. Shows detailed results for each problem
4. Saves the memory library to JSON

**Expected output:**
- Detailed hypotheses for each problem
- Cross-domain insights connecting AI and neuroscience
- Mechanism analysis for brain-related questions
- Experimental proposals for validation

### Option 2: Interactive Mode (Your Own Questions)

Ask your own research questions interactively:

```bash
python3 interactive_research.py
```

**Features:**
- Ask unlimited research questions
- View episodic memory statistics
- Save memory library at any time
- See example questions for inspiration

**Example session:**
```
> How can we reduce hallucinations in large language models?

🧠 RESEARCH PROBLEM: How can we reduce hallucinations in large language models?

📊 RESULTS:
   Outcome: SUCCESS
   Steps taken: 8

💡 HYPOTHESES GENERATED (3):
   1. Retrieval-Augmented Generation with Fact Verification...
   2. Uncertainty Quantification and Calibration...
   3. Multi-Model Consensus and Cross-Checking...

📚 KEY INSIGHTS:
   • Literature: Recent work on RAG systems, factuality metrics...
   • Mechanisms: Attention patterns in transformers, memory retrieval...
   • Synthesis: Combining neuroscience-inspired memory with AI fact-checking...
```

### Option 3: Python API (Programmatic Use)

Use in your own Python scripts:

```python
from neuro_ai_research_system import NeuroAIResearchSystem

# Initialize with real LLM
system = NeuroAIResearchSystem(use_real_llm=True)

# Build initial episodic memory
foundational_problems = [
    "How does attention work in transformers vs the brain?",
    "What are key differences between ANNs and biological neurons?",
    "How can neuroplasticity improve continual learning?"
]
system.build_episodic_library(foundational_problems)

# Solve your research problem
episode = system.solve_research_problem(
    "Design a brain-inspired architecture for few-shot learning",
    max_steps=15
)

# Access results
state = system.get_working_memory_state()
print("Hypotheses:", state['hypotheses'])
print("Insights:", state['context_notes'])

# Save memory library
system.save_memory_library("my_research_memory.json")
```

## Example Research Questions

### AI & Machine Learning
- How can we reduce hallucinations in large language models?
- Design a more efficient attention mechanism for transformers
- What are promising approaches to few-shot learning?
- How can we improve interpretability of deep neural networks?
- Create a novel approach to continual learning without catastrophic forgetting

### Neuroscience & Brain Function
- How does the brain process visual information hierarchically?
- What mechanisms enable rapid learning in the hippocampus?
- How do neural oscillations contribute to cognition?
- What is the role of glial cells in information processing?
- How does working memory capacity relate to prefrontal cortex function?

### Brain-Inspired AI
- How can spiking neural networks improve AI efficiency?
- Design an AI system inspired by the brain's predictive coding
- How can we implement working memory in neural networks?
- Create a continual learning system based on neuroplasticity
- Design a neuromorphic architecture for edge computing

### Medical Applications
- How can AI improve early detection of Alzheimer's disease?
- Design a brain-computer interface for paralysis patients
- What AI approaches can help in personalized medicine?
- How can deep learning assist in medical image diagnosis?
- Propose an AI system for real-time seizure prediction

### Cross-Domain Synthesis
- How can neuroscience insights improve AI architectures?
- What can AI teach us about brain function?
- Design a hybrid system combining symbolic AI with neural networks
- How can we create more energy-efficient AI using brain principles?
- Propose a unified theory connecting AI learning and synaptic plasticity

## Understanding the Output

### Episode Structure

Each problem-solving episode includes:

```
📊 RESULTS:
   Outcome: SUCCESS/PARTIAL/FAILURE
   Steps taken: 7

📝 KEY STEPS:
   1. DECOMPOSE_GOAL
   2. LITERATURE_SEARCH
   3. RETRIEVE_EPISODES
   4. RETRIEVE_KNOWLEDGE
   5. GENERATE_HYPOTHESES
   6. ANALYZE_MECHANISMS
   7. CROSS_DOMAIN_SYNTHESIS
   8. EVALUATE_HYPOTHESES
   9. PROPOSE_EXPERIMENTS
   10. REFINE_TOP_HYPOTHESIS

💡 HYPOTHESES GENERATED (3):
   [Detailed hypotheses with scientific reasoning]

📚 KEY INSIGHTS:
   [Cross-domain connections and mechanisms]
```

### What Each Action Does

- **DECOMPOSE_GOAL** - Breaks problem into subgoals
- **LITERATURE_SEARCH** - Identifies key concepts and recent research
- **RETRIEVE_EPISODES** - Pulls similar past problem-solving attempts
- **RETRIEVE_KNOWLEDGE** - Queries domain knowledge base
- **GENERATE_HYPOTHESES** - Creates 3-5 solution approaches
- **ANALYZE_MECHANISMS** - Deep dive into biological/computational mechanisms
- **CROSS_DOMAIN_SYNTHESIS** - Connects AI, neuroscience, medicine
- **EVALUATE_HYPOTHESES** - Critiques and ranks ideas
- **PROPOSE_EXPERIMENTS** - Suggests validation experiments
- **REFINE_TOP_HYPOTHESIS** - Improves best hypothesis with details

## Episodic Memory

### How It Works

1. **Storage**: Every problem solved is stored as an episode
2. **Retrieval**: When solving new problems, similar past episodes are retrieved
3. **Learning**: The system uses past experiences to inform new solutions
4. **Growth**: Performance improves as more episodes are added

### Memory Statistics

View your episodic memory:

```python
summary = system.get_research_summary()
print(f"Total problems solved: {summary['total_problems_solved']}")
print(f"Success rate: {summary['success_rate']*100:.1f}%")
print(f"Memory size: {summary['episodic_memory_size']} episodes")
```

### Saving and Loading

```python
# Save memory library
system.save_memory_library("my_memory.json")

# The JSON file contains:
# - All episodes with goals, steps, outcomes
# - Research log with timestamps
# - Full problem-solving traces
```

## Advanced Usage

### Custom Domain Knowledge

Add your own domain knowledge:

```python
system = NeuroAIResearchSystem(use_real_llm=True)

# Add custom knowledge
system.add_knowledge("Recent paper: Attention is All You Need (Vaswani et al., 2017)")
system.add_knowledge("Hippocampus has CA1, CA3, and DG regions")
system.add_knowledge("fMRI BOLD signal peaks 4-6 seconds after neural activity")

# Now solve problems with this knowledge
episode = system.solve_research_problem("Your question here")
```

### Adjusting Problem Complexity

Control how deeply the system explores:

```python
# Quick exploration (5-8 steps)
episode = system.solve_research_problem(problem, max_steps=8)

# Standard depth (10-15 steps)
episode = system.solve_research_problem(problem, max_steps=12)

# Deep exploration (15-20 steps)
episode = system.solve_research_problem(problem, max_steps=20)
```

### Batch Processing

Solve multiple problems:

```python
problems = [
    "Problem 1...",
    "Problem 2...",
    "Problem 3..."
]

results = []
for problem in problems:
    episode = system.solve_research_problem(problem)
    results.append({
        'problem': problem,
        'outcome': episode.outcome,
        'hypotheses': system.get_working_memory_state()['hypotheses']
    })

# Analyze results
for r in results:
    print(f"{r['problem']}: {r['outcome']}")
```

## Performance Tips

### 1. Build Initial Memory First

Start with foundational problems to build a strong episodic memory:

```python
foundational = [
    "Basic question 1",
    "Basic question 2",
    "Basic question 3"
]
system.build_episodic_library(foundational)

# Then tackle advanced problems
advanced_episode = system.solve_research_problem("Complex question")
```

### 2. Use Appropriate max_steps

- Simple questions: 8-10 steps
- Medium complexity: 12-15 steps
- Complex research: 15-20 steps

### 3. Add Relevant Knowledge

Pre-load domain knowledge for better results:

```python
# Add recent papers, facts, methodologies
system.add_knowledge("Your domain-specific knowledge")
```

### 4. Review Past Episodes

Learn from what worked:

```python
episodes = system.get_episodes()
successful = [e for e in episodes if e.outcome == 'success']

# Analyze successful patterns
for ep in successful:
    print(f"Goal: {ep.goal}")
    print(f"Steps: {[action for action, _ in ep.steps]}")
```

## Troubleshooting

### Issue: "GEMINI_API_KEY not found"

**Solution:**
```bash
export GEMINI_API_KEY='your-key-here'
```

### Issue: "google-generativeai not installed"

**Solution:**
```bash
pip install google-generativeai
# Or in virtual environment:
python3 -m venv venv && source venv/bin/activate && pip install openai
```

### Issue: API rate limits

**Solution:**
- Reduce max_steps to make fewer LLM calls
- Add delays between problems
- Use a higher-tier OpenAI plan

### Issue: Poor quality outputs

**Solution:**
- Build episodic memory first with foundational problems
- Add more domain knowledge
- Increase max_steps for deeper exploration
- Try different problem phrasings

## Cost Considerations

Using Google Gemini:
- **Free Tier:** 15 requests/min, 1,500 requests/day
- Each problem: ~5-15 API calls depending on max_steps
- Cost per problem: **FREE** (within quota) or ~$0.01-0.05 (paid)
- Building episodic library (5 problems): **FREE** (within quota)

**Gemini Advantages:**
- Generous free tier for research
- Much cheaper than GPT-4 if you exceed free quota
- Fast response times with flash models
- Long context windows (up to 1M tokens)

**Tips to reduce costs:**
- Stay within free tier limits (1,500 requests/day)
- Use gemini-2.0-flash-exp (fastest, free)
- Reduce max_steps
- Build memory library once, reuse it
- Use MockLLM for testing/development

## What Makes This System Unique

### Traditional LLM Usage
```
User: "How can we improve AI?"
LLM: [Single response]
```

### This Cognitive System
```
1. Decomposes goal into subgoals
2. Searches literature for context
3. Retrieves similar past solutions
4. Generates multiple hypotheses
5. Analyzes mechanisms
6. Synthesizes cross-domain insights
7. Evaluates approaches
8. Proposes experiments
9. Refines best hypothesis
10. Stores episode for future learning
```

**Result:** Much more thorough, creative, and scientifically rigorous solutions!

## Next Steps

1. **Run the demo**: `python3 neuro_ai_research_system.py`
2. **Try interactive mode**: `python3 interactive_research.py`
3. **Ask your own questions** in AI/medical/brain domains
4. **Build your episodic library** with foundational problems
5. **Tackle advanced research questions** using accumulated memory
6. **Save your memory library** for future sessions

## Support

For issues or questions:
- Check this guide for solutions
- Review the cognitive_system.py implementation
- Examine example outputs from the demo
- Modify the controller for your specific needs

## Citation

This system implements the cognitive architecture described in the prototype document, with specialized extensions for AI, neuroscience, and medical research.
