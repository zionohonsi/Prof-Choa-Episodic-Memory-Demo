# Neuro-AI Research System - Quick Reference

## 🚀 Quick Start

### Setup (One-time)
```bash
# 1. Set API key
export GEMINI_API_KEY='your-key-here'

# 2. Install Google Generative AI (if needed)
pip install google-generativeai
```

**Get API key:** https://aistudio.google.com/app/apikey

### Run Demo
```bash
python3 neuro_ai_research_system.py
```

### Interactive Mode
```bash
python3 interactive_research.py
```

## 📋 Common Commands

### Python API
```python
from neuro_ai_research_system import NeuroAIResearchSystem

# Initialize
system = NeuroAIResearchSystem(use_real_llm=True)

# Build memory
problems = ["Question 1", "Question 2", "Question 3"]
system.build_episodic_library(problems)

# Solve problem
episode = system.solve_research_problem("Your question", max_steps=15)

# Get results
state = system.get_working_memory_state()
print(state['hypotheses'])

# Save memory
system.save_memory_library("memory.json")
```

## 🎯 Example Questions

**AI/ML:**
- How can we reduce hallucinations in LLMs?
- Design efficient attention for transformers
- Improve few-shot learning approaches

**Neuroscience:**
- How does hippocampus enable rapid learning?
- What role do neural oscillations play?
- How does working memory work?

**Brain-Inspired AI:**
- Design spiking neural network for efficiency
- Implement predictive coding in AI
- Create neuroplasticity-based continual learning

**Medical:**
- AI for early Alzheimer's detection
- Brain-computer interface design
- Deep learning for medical imaging

## 🔧 Key Parameters

```python
# Control exploration depth
max_steps=8   # Quick (5-8 steps)
max_steps=12  # Standard (10-15 steps)
max_steps=20  # Deep (15-20 steps)

# Use real or mock LLM
use_real_llm=True   # OpenAI GPT-4
use_real_llm=False  # MockLLM (testing)
```

## 📊 Understanding Output

```
📊 RESULTS:
   Outcome: SUCCESS/PARTIAL/FAILURE
   Steps taken: X

📝 KEY STEPS:
   [Actions taken by system]

💡 HYPOTHESES GENERATED:
   [Solution approaches]

📚 KEY INSIGHTS:
   [Cross-domain connections]
```

## 🧠 System Actions

1. **DECOMPOSE_GOAL** - Break into subgoals
2. **LITERATURE_SEARCH** - Find key concepts
3. **RETRIEVE_EPISODES** - Use past experience
4. **RETRIEVE_KNOWLEDGE** - Query knowledge base
5. **GENERATE_HYPOTHESES** - Create solutions
6. **ANALYZE_MECHANISMS** - Deep mechanism analysis
7. **CROSS_DOMAIN_SYNTHESIS** - Connect AI/neuro/med
8. **EVALUATE_HYPOTHESES** - Critique ideas
9. **PROPOSE_EXPERIMENTS** - Suggest validation
10. **REFINE_TOP_HYPOTHESIS** - Improve best idea

## 💡 Tips

✅ **Build memory first** - Start with foundational problems
✅ **Use appropriate steps** - More steps = deeper exploration
✅ **Add domain knowledge** - Pre-load relevant facts
✅ **Save your memory** - Reuse episodic library

## ⚠️ Troubleshooting

**API key not found:**
```bash
export GEMINI_API_KEY='your-key'
```

**Gemini library not installed:**
```bash
pip install google-generativeai
```

**Virtual environment needed:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install google-generativeai
```

## 📁 Files

- `neuro_ai_research_system.py` - Main system (automated demo)
- `interactive_research.py` - Interactive Q&A mode
- `NEURO_AI_GUIDE.md` - Complete documentation
- `cognitive_architecture/` - Core implementation

## 🎓 What Makes It Special

**Traditional LLM:** Single response
**This System:** 
- Decomposes problems
- Searches literature
- Uses past experience
- Generates multiple hypotheses
- Analyzes mechanisms
- Synthesizes cross-domain
- Proposes experiments
- Learns over time

## 📈 Performance

- **Cost per problem:** FREE (within quota) or ~$0.01-0.05 (Gemini)
- **Time per problem:** 20-60 seconds (fast!)
- **Quality:** Research-grade insights
- **Learning:** Improves with each problem
- **Free tier:** 1,500 requests/day

## 🔗 Quick Links

- Full Guide: `NEURO_AI_GUIDE.md`
- Core System: `cognitive_architecture/README.md`
- Quick Start: `cognitive_architecture/QUICKSTART.md`
- Tests: `cognitive_architecture/test_cognitive_system.py`

---

**Ready to start?** Run: `python3 interactive_research.py`
