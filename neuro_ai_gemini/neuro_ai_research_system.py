"""
Neuro-AI Research System
A specialized cognitive architecture for solving AI, medical, and brain-related problems.
Builds episodic memory and tackles progressively advanced problems using real LLMs.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

# Import the cognitive system
sys.path.insert(0, '/home/ubuntu/cognitive_architecture')
from cognitive_system import (
    CognitiveSystem, Controller, Episode, 
    WorkingMemory, EpisodicMemory, SemanticMemory, SkillCore
)


# ============================================================================
# Real LLM Integration
# ============================================================================

class GeminiLLM:
    """Real Gemini LLM integration."""
    
    def __init__(self, model="gemini-2.0-flash-exp"):
        try:
            # Try new google.genai package first
            try:
                from google import genai
                from google.genai import types
                use_new_api = True
            except ImportError:
                # Fall back to old package
                import google.generativeai as genai
                use_new_api = False
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("⚠️  GEMINI_API_KEY not found in environment")
                self.available = False
                return
            
            if use_new_api:
                # New API
                self.client = genai.Client(api_key=api_key)
                self.model_name = model
                self.use_new_api = True
            else:
                # Old API
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model)
                self.use_new_api = False
            
            self.available = True
            print(f"✓ Gemini API initialized with model: {model}")
        except ImportError:
            print("⚠️  Google Generative AI library not installed.")
            print("   Run: pip install google-generativeai")
            self.available = False
        except Exception as e:
            print(f"⚠️  Gemini initialization failed: {e}")
            self.available = False
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Gemini API."""
        if not self.available:
            return "Gemini not available. Please install and configure."
        
        try:
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            if self.use_new_api:
                # New API
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=800,
                    )
                )
                return response.text
            else:
                # Old API
                response = self.model.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': 0.7,
                        'max_output_tokens': 800,
                    }
                )
                return response.text
        except Exception as e:
            return f"Error calling Gemini: {e}"


# ============================================================================
# Specialized Controller for Neuro-AI Research
# ============================================================================

class NeuroAIController(Controller):
    """Enhanced controller for AI, medical, and brain research."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_actions = [
            "LITERATURE_SEARCH",
            "ANALYZE_MECHANISMS",
            "PROPOSE_EXPERIMENTS",
            "CROSS_DOMAIN_SYNTHESIS"
        ]
    
    def select_action(self) -> str:
        """Enhanced action selection for research tasks."""
        wm = self.working_mem
        trace_actions = [action for action, _ in self.trace]
        
        # Initial decomposition
        if not wm.subgoals:
            return "DECOMPOSE_GOAL"
        
        # Early literature search for research problems
        if len(trace_actions) == 1 and "research" in wm.goal.lower():
            return "LITERATURE_SEARCH"
        
        # Retrieve past episodes early
        if len(trace_actions) == 2:
            return "RETRIEVE_EPISODES"
        
        # Retrieve domain knowledge
        if len(trace_actions) == 3:
            return "RETRIEVE_KNOWLEDGE"
        
        # Generate hypotheses
        if not wm.hypotheses:
            return "GENERATE_HYPOTHESES"
        
        # Analyze mechanisms for brain/medical topics
        if ("brain" in wm.goal.lower() or "neural" in wm.goal.lower()) and \
           "ANALYZE_MECHANISMS" not in trace_actions:
            return "ANALYZE_MECHANISMS"
        
        # Cross-domain synthesis for complex problems
        if len(wm.hypotheses) > 2 and "CROSS_DOMAIN_SYNTHESIS" not in trace_actions:
            return "CROSS_DOMAIN_SYNTHESIS"
        
        # Evaluate hypotheses
        if wm.hypotheses and not wm.intermediate_results:
            return "EVALUATE_HYPOTHESES"
        
        # Propose experiments for validation
        if wm.intermediate_results and "PROPOSE_EXPERIMENTS" not in trace_actions:
            return "PROPOSE_EXPERIMENTS"
        
        # Refine top hypothesis
        return "REFINE_TOP_HYPOTHESIS"
    
    def execute_action(self, action: str) -> Any:
        """Execute domain-specific actions."""
        wm = self.working_mem
        
        if action == "LITERATURE_SEARCH":
            prompt = f"""You are a research assistant. Identify key concepts, recent advances, and important papers related to:

Goal: {wm.goal}

Provide:
1. Key concepts and terminology
2. Recent breakthrough findings (2020-2025)
3. Important research directions
4. Relevant methodologies

Format as a concise research summary."""
            out = self.skill_core.call("research_assistant", prompt)
            return out
        
        elif action == "ANALYZE_MECHANISMS":
            context = self._build_context()
            prompt = f"""You are a neuroscience expert. Analyze the biological/computational mechanisms involved in:

Goal: {wm.goal}

Context: {context}

Provide:
1. Underlying biological mechanisms
2. Computational principles
3. Key neural circuits or pathways involved
4. Connections to AI/ML approaches

Be specific and scientifically rigorous."""
            out = self.skill_core.call("neuroscience_expert", prompt)
            return out
        
        elif action == "PROPOSE_EXPERIMENTS":
            hypotheses_text = '\n'.join([f"{i+1}. {h}" for i, h in enumerate(wm.hypotheses[:3])])
            prompt = f"""You are an experimental scientist. Propose concrete experiments to validate these hypotheses:

Goal: {wm.goal}

Top Hypotheses:
{hypotheses_text}

For each hypothesis, propose:
1. Experimental design
2. Required data/resources
3. Expected outcomes
4. Validation criteria

Be practical and specific."""
            out = self.skill_core.call("experimental_scientist", prompt)
            return out
        
        elif action == "CROSS_DOMAIN_SYNTHESIS":
            hypotheses_text = '\n'.join([f"{i+1}. {h}" for i, h in enumerate(wm.hypotheses)])
            context = self._build_context()
            prompt = f"""You are a cross-domain researcher. Synthesize insights from AI, neuroscience, and medicine:

Goal: {wm.goal}

Current Hypotheses:
{hypotheses_text}

Context: {context}

Identify:
1. Cross-domain connections and analogies
2. How AI/ML can inform neuroscience/medicine
3. How brain science can improve AI
4. Novel hybrid approaches

Provide integrative insights."""
            out = self.skill_core.call("cross_domain_synthesizer", prompt)
            return out
        
        # Fall back to base actions
        return super().execute_action(action)
    
    def update_working_memory(self, action: str, result: Any):
        """Enhanced memory updates for research actions."""
        if action == "LITERATURE_SEARCH":
            self.working_mem.context_notes.append(f"Literature: {str(result)[:200]}...")
        elif action == "ANALYZE_MECHANISMS":
            self.working_mem.context_notes.append(f"Mechanisms: {str(result)[:200]}...")
        elif action == "PROPOSE_EXPERIMENTS":
            self.working_mem.intermediate_results.append(result)
        elif action == "CROSS_DOMAIN_SYNTHESIS":
            self.working_mem.context_notes.append(f"Synthesis: {str(result)[:200]}...")
        else:
            super().update_working_memory(action, result)


# ============================================================================
# Specialized Research System
# ============================================================================

class NeuroAIResearchSystem(CognitiveSystem):
    """Specialized system for AI/medical/brain research."""
    
    def __init__(self, use_real_llm=True):
        # Initialize with real or mock LLM
        if use_real_llm:
            llm = GeminiLLM()
            if not llm.available:
                print("⚠️  Falling back to MockLLM")
                from cognitive_system import MockLLM
                llm = MockLLM()
        else:
            from cognitive_system import MockLLM
            llm = MockLLM()
        
        # Initialize base system
        super().__init__(llm=llm)
        
        # Replace controller with specialized version
        self.controller = NeuroAIController(
            self.skill_core,
            self.working_mem,
            self.episodic_mem,
            self.semantic_mem
        )
        
        # Add domain knowledge
        self._initialize_domain_knowledge()
        
        # Track research progress
        self.research_log = []
    
    def _initialize_domain_knowledge(self):
        """Initialize with AI/medical/brain domain knowledge."""
        knowledge = [
            "The brain uses sparse, distributed representations for efficient information processing",
            "Transformers and attention mechanisms are inspired by selective attention in the brain",
            "Neuroplasticity allows the brain to reorganize and adapt through learning",
            "Deep learning models share computational principles with hierarchical processing in visual cortex",
            "Reinforcement learning in AI is based on dopaminergic reward prediction in the brain",
            "Working memory capacity is limited to 4-7 items (Cowan's law)",
            "The hippocampus is critical for episodic memory formation and spatial navigation",
            "Predictive coding suggests the brain constantly generates predictions about sensory input",
            "Meta-learning (learning to learn) has parallels in prefrontal cortex function",
            "Spiking neural networks more closely model biological neurons than artificial neural networks",
            "The default mode network is active during rest and self-referential thinking",
            "Neuromorphic computing aims to replicate brain-like efficiency in hardware",
            "fMRI measures blood oxygen levels as a proxy for neural activity",
            "Optogenetics allows precise control of neurons using light",
            "Brain-computer interfaces can decode neural signals for prosthetic control"
        ]
        
        for fact in knowledge:
            self.add_knowledge(fact)
    
    def solve_research_problem(self, problem: str, max_steps: int = 15) -> Episode:
        """Solve a research problem and log the process."""
        print(f"\n{'='*70}")
        print(f"🧠 RESEARCH PROBLEM: {problem}")
        print(f"{'='*70}\n")
        
        # Solve using the cognitive system
        episode = self.solve(problem, max_steps=max_steps)
        
        # Log the research
        self.research_log.append({
            'timestamp': datetime.now().isoformat(),
            'problem': problem,
            'outcome': episode.outcome,
            'steps': len(episode.steps),
            'hypotheses_generated': len(self.working_mem.hypotheses)
        })
        
        # Display results
        self._display_results(episode)
        
        return episode
    
    def _display_results(self, episode: Episode):
        """Display research results in a readable format."""
        print(f"\n📊 RESULTS:")
        print(f"   Outcome: {episode.outcome.upper()}")
        print(f"   Steps taken: {len(episode.steps)}")
        
        # Show key steps
        print(f"\n📝 KEY STEPS:")
        for i, (action, result) in enumerate(episode.steps, 1):
            print(f"   {i}. {action}")
        
        # Show hypotheses
        wm_state = self.get_working_memory_state()
        if wm_state['hypotheses']:
            print(f"\n💡 HYPOTHESES GENERATED ({len(wm_state['hypotheses'])}):")
            for i, hyp in enumerate(wm_state['hypotheses'][:3], 1):
                print(f"\n   {i}. {hyp[:300]}...")
        
        # Show context notes
        if wm_state['context_notes']:
            print(f"\n📚 KEY INSIGHTS:")
            for note in wm_state['context_notes'][-3:]:
                print(f"   • {note[:200]}...")
        
        print(f"\n{'='*70}\n")
    
    def build_episodic_library(self, problems: List[str], max_steps: int = 12):
        """Build episodic memory by solving multiple problems."""
        print(f"\n{'='*70}")
        print(f"🏗️  BUILDING EPISODIC MEMORY LIBRARY")
        print(f"{'='*70}")
        print(f"Solving {len(problems)} problems to build experience...\n")
        
        for i, problem in enumerate(problems, 1):
            print(f"[{i}/{len(problems)}] {problem}")
            episode = self.solve_research_problem(problem, max_steps=max_steps)
            print(f"   ✓ Completed: {episode.outcome}\n")
        
        # Summary
        episodes = self.get_episodes()
        print(f"\n{'='*70}")
        print(f"📚 EPISODIC MEMORY LIBRARY BUILT")
        print(f"{'='*70}")
        print(f"Total episodes: {len(episodes)}")
        print(f"Successful: {sum(1 for e in episodes if e.outcome == 'success')}")
        print(f"Partial: {sum(1 for e in episodes if e.outcome == 'partial')}")
        print(f"\nThe system can now leverage these experiences for more advanced problems!")
        print(f"{'='*70}\n")
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of all research conducted."""
        episodes = self.get_episodes()
        return {
            'total_problems_solved': len(episodes),
            'success_rate': sum(1 for e in episodes if e.outcome == 'success') / len(episodes) if episodes else 0,
            'total_hypotheses_generated': sum(len(self.working_mem.hypotheses) for _ in episodes),
            'research_log': self.research_log,
            'episodic_memory_size': len(episodes)
        }
    
    def save_memory_library(self, filepath: str = "episodic_memory_library.json"):
        """Save episodic memory to file."""
        episodes = self.get_episodes()
        data = {
            'saved_at': datetime.now().isoformat(),
            'total_episodes': len(episodes),
            'episodes': [ep.to_dict() for ep in episodes],
            'research_log': self.research_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Episodic memory library saved to: {filepath}")


# ============================================================================
# Main Demonstration
# ============================================================================

def main():
    """Main demonstration of the Neuro-AI Research System."""
    
    print("\n" + "="*70)
    print(" NEURO-AI RESEARCH SYSTEM")
    print(" Specialized Cognitive Architecture for AI/Medical/Brain Research")
    print("="*70)
    
    # Check for Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  GEMINI_API_KEY not set.")
        print("   Set it with: export GEMINI_API_KEY='your-key-here'")
        print("   Or the system will use MockLLM for demonstration.\n")
        use_real = input("Continue with MockLLM? (y/n): ").lower().strip() == 'y'
        if not use_real:
            print("Exiting. Please set GEMINI_API_KEY and try again.")
            return
        use_real_llm = False
    else:
        use_real_llm = True
        print("\n✓ Gemini API key found. Using real LLM.\n")
    
    # Initialize system
    system = NeuroAIResearchSystem(use_real_llm=use_real_llm)
    
    # Phase 1: Build episodic memory with foundational problems
    print("\n" + "="*70)
    print(" PHASE 1: BUILDING EPISODIC MEMORY")
    print("="*70)
    
    foundational_problems = [
        "How does attention mechanism in transformers relate to biological attention in the brain?",
        "What are the key differences between artificial and biological neural networks?",
        "How can neuroplasticity principles improve continual learning in AI?",
        "What role does the hippocampus play in memory consolidation?",
        "How can reinforcement learning be improved using insights from dopamine signaling?"
    ]
    
    system.build_episodic_library(foundational_problems, max_steps=10)
    
    # Phase 2: Tackle advanced problems using episodic memory
    print("\n" + "="*70)
    print(" PHASE 2: SOLVING ADVANCED PROBLEMS")
    print("="*70)
    print("Now leveraging episodic memory for more complex problems...\n")
    
    advanced_problems = [
        "Design a novel neural architecture that combines transformer attention with hippocampal memory mechanisms for improved episodic learning",
        "Propose a brain-inspired approach to catastrophic forgetting in continual learning systems",
        "How can we create more energy-efficient AI by mimicking neuromorphic principles from the brain?"
    ]
    
    for problem in advanced_problems:
        system.solve_research_problem(problem, max_steps=15)
        input("\nPress Enter to continue to next problem...")
    
    # Summary
    print("\n" + "="*70)
    print(" RESEARCH SUMMARY")
    print("="*70)
    
    summary = system.get_research_summary()
    print(f"\nTotal problems solved: {summary['total_problems_solved']}")
    print(f"Success rate: {summary['success_rate']*100:.1f}%")
    print(f"Episodic memory size: {summary['episodic_memory_size']} episodes")
    
    # Save memory library
    system.save_memory_library("/home/ubuntu/neuro_ai_memory_library.json")
    
    print("\n" + "="*70)
    print(" SYSTEM READY FOR YOUR RESEARCH QUESTIONS")
    print("="*70)
    print("\nYou can now:")
    print("  • Ask the system to solve your own AI/medical/brain problems")
    print("  • The system will use its episodic memory to inform solutions")
    print("  • Each problem solved adds to the memory library")
    print("  • The system gets better with more experience!")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
