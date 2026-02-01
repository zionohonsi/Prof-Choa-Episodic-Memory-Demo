# Gemini API Setup Guide

## Quick Setup

### 1. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Get API Key" or "Create API Key"
3. Copy your API key

### 2. Install Required Package

```bash
pip install google-generativeai
```

Or with virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install google-generativeai
```

### 3. Set API Key

**Linux/Mac:**
```bash
export GEMINI_API_KEY='your-api-key-here'
```

**Permanent (add to ~/.bashrc or ~/.zshrc):**
```bash
echo "export GEMINI_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

### 4. Verify Setup

```bash
python3 -c "import os; print('✓ API key set' if os.getenv('GEMINI_API_KEY') else '✗ API key not found')"
```

## Run the System

### Automated Demo
```bash
python3 neuro_ai_research_system.py
```

### Interactive Mode
```bash
python3 interactive_research.py
```

## Gemini Models Available

The system uses **gemini-2.0-flash-exp** by default (fast and capable).

Other options:
- `gemini-2.0-flash-exp` - Fast, experimental (default)
- `gemini-1.5-pro` - Most capable, slower
- `gemini-1.5-flash` - Fast, efficient

To change model, edit in `neuro_ai_research_system.py`:
```python
class GeminiLLM:
    def __init__(self, model="gemini-1.5-pro"):  # Change here
```

## Advantages of Gemini

✅ **Free tier available** - Generous free quota
✅ **Fast responses** - Especially with flash models
✅ **Long context** - Up to 1M tokens (gemini-1.5-pro)
✅ **Multimodal** - Can handle images, video, audio
✅ **Cost-effective** - Lower cost than GPT-4

## Cost Comparison

**Gemini 2.0 Flash (Free Tier):**
- 15 requests per minute
- 1,500 requests per day
- Perfect for research and experimentation

**Gemini 1.5 Pro (Paid):**
- Input: $1.25 per 1M tokens
- Output: $5.00 per 1M tokens
- Much cheaper than GPT-4

**Estimated Cost per Research Problem:**
- With Gemini Flash: **FREE** (within quota)
- With Gemini Pro: ~$0.01-0.05 per problem

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
```

### Issue: "externally-managed-environment"

**Solution:** Use virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install google-generativeai
```

### Issue: API quota exceeded

**Solution:**
- Wait for quota to reset (daily/minute limits)
- Upgrade to paid tier
- Reduce max_steps to make fewer API calls

### Issue: Rate limit errors

**Solution:**
- Reduce number of problems solved at once
- Add delays between problems
- Use lower rate limit models

## Testing the Integration

Quick test:
```python
import os
import google.generativeai as genai

# Configure
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Test
model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content("Say hello!")
print(response.text)
```

## Example Usage with Gemini

```python
from neuro_ai_research_system import NeuroAIResearchSystem

# Initialize with Gemini
system = NeuroAIResearchSystem(use_real_llm=True)

# Solve a problem
episode = system.solve_research_problem(
    "How can we reduce hallucinations in large language models?",
    max_steps=15
)

# View results
state = system.get_working_memory_state()
print("Hypotheses:", state['hypotheses'])
```

## Performance Tips

1. **Use Flash models** for faster responses
2. **Adjust max_steps** based on complexity (8-15 typical)
3. **Build episodic memory** first for better results
4. **Monitor quota** if using free tier

## Getting Help

- Gemini API Docs: https://ai.google.dev/docs
- API Key Management: https://aistudio.google.com/app/apikey
- Pricing: https://ai.google.dev/pricing

## Ready to Start!

Once your API key is set:
```bash
export GEMINI_API_KEY='your-key-here'
python3 neuro_ai_research_system.py
```

Enjoy exploring AI, neuroscience, and medical research with Gemini! 🧠✨
