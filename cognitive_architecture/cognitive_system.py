"""
Cognitive Architecture: Working Memory System with LLM-based Skill Engine
A minimal implementation of a learning, simulating, and inventing system.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WorkingMemory:
    """Dynamic scratchpad for current reasoning process."""
    goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    intermediate_results: List[Any] = field(default_factory=list)
    context_notes: List[str] = field(default_factory=list)
    
    def clear(self):
        """Reset working memory."""
        self.goal = None
        self.subgoals = []
        self.hypotheses = []
        self.intermediate_results = []
        self.context_notes = []


@dataclass
class Episode:
    """Record of a single problem-solving attempt."""
    goal: str
    steps: List[Tuple[str, Any]]
    outcome: str  # "success", "failure", "partial"
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for serialization."""
        return {
            "goal": self.goal,
            "steps": [(action, str(result)) for action, result in self.steps],
            "outcome": self.outcome,
            "notes": self.notes
        }


class EpisodicMemory:
    """Store and retrieve past problem-solving experiences."""
    
    def __init__(self):
        self.episodes: List[Episode] = []
    
    def store_episode(self, episode: Episode):
        """Store a new episode."""
        self.episodes.append(episode)
    
    def retrieve_similar(self, goal: str, k: int = 3) -> List[Episode]:
        """Retrieve k most similar episodes based on goal similarity."""
        if not self.episodes:
            return []
        
        # Simple keyword-based similarity (can be enhanced with embeddings)
        def similarity_score(episode: Episode) -> float:
            goal_words = set(goal.lower().split())
            episode_words = set(episode.goal.lower().split())
            if not goal_words or not episode_words:
                return 0.0
            intersection = goal_words.intersection(episode_words)
            union = goal_words.union(episode_words)
            return len(intersection) / len(union)
        
        ranked = sorted(self.episodes, key=similarity_score, reverse=True)
        return ranked[:k]
    
    def get_all_episodes(self) -> List[Episode]:
        """Return all stored episodes."""
        return self.episodes


class SemanticMemory:
    """Store and retrieve factual knowledge."""
    
    def __init__(self, knowledge_base: Optional[List[str]] = None):
        self.knowledge_base = knowledge_base or []
    
    def add_knowledge(self, knowledge: str):
        """Add new knowledge to the base."""
        self.knowledge_base.append(knowledge)
    
    def retrieve_relevant(self, query: str, k: int = 5) -> List[str]:
        """Retrieve k most relevant knowledge items."""
        if not self.knowledge_base:
            return []
        
        # Simple keyword-based search
        def relevance_score(knowledge: str) -> float:
            query_words = set(query.lower().split())
            knowledge_words = set(knowledge.lower().split())
            if not query_words or not knowledge_words:
                return 0.0
            intersection = query_words.intersection(knowledge_words)
            return len(intersection)
        
        ranked = sorted(self.knowledge_base, key=relevance_score, reverse=True)
        return ranked[:k]


class SkillCore:
    """Wrapper around LLM for skill execution."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Execute LLM call with system and user prompts."""
        return self.llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)


# ============================================================================
# Controller and Main Loop
# ============================================================================

# Define available actions
ACTIONS = [
    "DECOMPOSE_GOAL",
    "RETRIEVE_EPISODES",
    "RETRIEVE_KNOWLEDGE",
    "GENERATE_HYPOTHESES",
    "EVALUATE_HYPOTHESES",
    "REFINE_TOP_HYPOTHESIS",
    "SUMMARIZE_PROGRESS"
]


class Controller:
    """Orchestrates the cognitive system, selecting and executing actions."""
    
    def __init__(self, skill_core: SkillCore, working_mem: WorkingMemory,
                 episodic_mem: EpisodicMemory, semantic_mem: SemanticMemory):
        self.skill_core = skill_core
        self.working_mem = working_mem
        self.episodic_mem = episodic_mem
        self.semantic_mem = semantic_mem
        self.trace: List[Tuple[str, Any]] = []
    
    def set_goal(self, goal_text: str):
        """Set the current goal in working memory."""
        self.working_mem.goal = goal_text
        self.trace = []
    
    def select_action(self) -> str:
        """
        Select next action based on current state (heuristic-based).
        Can be replaced with learned policy later.
        """
        wm = self.working_mem
        
        # If no subgoals, decompose first
        if not wm.subgoals:
            return "DECOMPOSE_GOAL"
        
        # If no hypotheses, generate them
        if not wm.hypotheses:
            return "GENERATE_HYPOTHESES"
        
        # If hypotheses exist but not evaluated, evaluate
        if len(wm.hypotheses) > 0 and not wm.intermediate_results:
            return "EVALUATE_HYPOTHESES"
        
        # Periodically retrieve knowledge and episodes
        if len(self.trace) in (1, 4):
            return "RETRIEVE_KNOWLEDGE"
        
        if len(self.trace) == 2:
            return "RETRIEVE_EPISODES"
        
        # Default: refine top hypothesis
        return "REFINE_TOP_HYPOTHESIS"
    
    def execute_action(self, action: str) -> Any:
        """Execute the selected action using LLM or memory systems."""
        wm = self.working_mem
        
        if action == "DECOMPOSE_GOAL":
            prompt = f"""You are a planner. Decompose this goal into 3-7 concrete subgoals:

Goal: {wm.goal}

Provide the subgoals as a numbered list."""
            out = self.skill_core.call(system_prompt="planner", user_prompt=prompt)
            return self._parse_subgoals(out)
        
        elif action == "RETRIEVE_EPISODES":
            episodes = self.episodic_mem.retrieve_similar(wm.goal)
            return episodes
        
        elif action == "RETRIEVE_KNOWLEDGE":
            knowledge = self.semantic_mem.retrieve_relevant(wm.goal)
            return knowledge
        
        elif action == "GENERATE_HYPOTHESES":
            context = self._build_context()
            prompt = f"""You are an inventor. Generate 3-5 creative hypotheses or solution approaches for this goal:

Goal: {wm.goal}

Subgoals: {', '.join(wm.subgoals)}

Context: {context}

Provide diverse, innovative approaches."""
            out = self.skill_core.call(system_prompt="inventor", user_prompt=prompt)
            return self._parse_hypotheses(out)
        
        elif action == "EVALUATE_HYPOTHESES":
            hypotheses_text = '\n'.join([f"{i+1}. {h}" for i, h in enumerate(wm.hypotheses)])
            prompt = f"""You are a critic. Evaluate these hypotheses for the goal:

Goal: {wm.goal}

Hypotheses:
{hypotheses_text}

Rank them by feasibility, creativity, and potential impact. Provide scores and reasoning."""
            out = self.skill_core.call(system_prompt="critic", user_prompt=prompt)
            return self._parse_evaluation(out)
        
        elif action == "REFINE_TOP_HYPOTHESIS":
            if not wm.hypotheses:
                return "No hypothesis to refine"
            
            top_hypothesis = wm.hypotheses[0]
            prompt = f"""You are a refiner. Improve and elaborate on this hypothesis:

Goal: {wm.goal}

Current Best Hypothesis: {top_hypothesis}

Provide a refined, more detailed version with concrete steps or mechanisms."""
            out = self.skill_core.call(system_prompt="refiner", user_prompt=prompt)
            return self._parse_refined_hypothesis(out)
        
        elif action == "SUMMARIZE_PROGRESS":
            prompt = f"""Summarize the current progress on this goal:

Goal: {wm.goal}
Subgoals: {', '.join(wm.subgoals)}
Hypotheses explored: {len(wm.hypotheses)}
Current notes: {'; '.join(wm.context_notes)}

Provide a concise summary of what has been accomplished."""
            out = self.skill_core.call(system_prompt="summarizer", user_prompt=prompt)
            return out
        
        return None
    
    def update_working_memory(self, action: str, result: Any):
        """Update working memory based on action result."""
        wm = self.working_mem
        
        if action == "DECOMPOSE_GOAL":
            wm.subgoals = result
        
        elif action == "RETRIEVE_EPISODES":
            if result:
                summary = f"Similar episodes: {len(result)} found"
                wm.context_notes.append(summary)
        
        elif action == "RETRIEVE_KNOWLEDGE":
            if result:
                summary = f"Key facts: {len(result)} items retrieved"
                wm.context_notes.append(summary)
        
        elif action == "GENERATE_HYPOTHESES":
            wm.hypotheses.extend(result)
        
        elif action == "EVALUATE_HYPOTHESES":
            wm.intermediate_results.append(result)
        
        elif action == "REFINE_TOP_HYPOTHESIS":
            if wm.hypotheses:
                wm.hypotheses[0] = result
        
        elif action == "SUMMARIZE_PROGRESS":
            wm.context_notes.append(result)
    
    def step(self):
        """Execute one reasoning step."""
        action = self.select_action()
        result = self.execute_action(action)
        self.update_working_memory(action, result)
        self.trace.append((action, result))
        return action, result
    
    def goal_satisfied(self) -> bool:
        """Check if goal is satisfied."""
        # Naive: if we have hypotheses and evaluations
        return (len(self.working_mem.hypotheses) > 0 and 
                len(self.working_mem.intermediate_results) > 0)
    
    def stuck(self) -> bool:
        """Check if system is stuck."""
        # Simple: too many steps without progress
        return False  # Can be elaborated
    
    def finalize_episode(self) -> Episode:
        """Create and store episode from current trace."""
        outcome = "success" if self.goal_satisfied() else "partial"
        episode = Episode(
            goal=self.working_mem.goal,
            steps=self.trace,
            outcome=outcome,
            notes=self.working_mem.context_notes
        )
        self.episodic_mem.store_episode(episode)
        return episode
    
    def run_until_done(self, max_steps: int = 20) -> Episode:
        """Run the cognitive loop until goal is satisfied or max steps reached."""
        for step_num in range(max_steps):
            if self.goal_satisfied() or self.stuck():
                break
            self.step()
        
        return self.finalize_episode()
    
    # Helper methods for parsing LLM outputs
    
    def _build_context(self) -> str:
        """Build context string from working memory."""
        if not self.working_mem.context_notes:
            return "No prior context"
        return "; ".join(self.working_mem.context_notes)
    
    def _parse_subgoals(self, text: str) -> List[str]:
        """Parse subgoals from LLM output."""
        lines = text.strip().split('\n')
        subgoals = []
        for line in lines:
            line = line.strip()
            # Remove numbering like "1.", "2)", etc.
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove leading numbering/bullets
                cleaned = line.lstrip('0123456789.-*) ').strip()
                if cleaned:
                    subgoals.append(cleaned)
        return subgoals if subgoals else [text.strip()]
    
    def _parse_hypotheses(self, text: str) -> List[str]:
        """Parse hypotheses from LLM output."""
        return self._parse_subgoals(text)  # Similar parsing logic
    
    def _parse_evaluation(self, text: str) -> str:
        """Parse evaluation from LLM output."""
        return text.strip()
    
    def _parse_refined_hypothesis(self, text: str) -> str:
        """Parse refined hypothesis from LLM output."""
        return text.strip()


# ============================================================================
# Mock LLM for Testing (replace with real LLM)
# ============================================================================

class MockLLM:
    """Mock LLM for testing purposes."""
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate mock responses based on system prompt."""
        if system_prompt == "planner":
            return """1. Research existing approaches
2. Identify key constraints and requirements
3. Brainstorm novel solutions
4. Prototype the most promising idea
5. Test and validate the solution"""
        
        elif system_prompt == "inventor":
            return """1. Hybrid approach combining neural networks with symbolic reasoning
2. Meta-learning system that learns to learn from fewer examples
3. Evolutionary algorithm with novelty search to explore solution space"""
        
        elif system_prompt == "critic":
            return """Hypothesis 1: Score 8/10 - Feasible and innovative
Hypothesis 2: Score 9/10 - Most promising, balances creativity and practicality
Hypothesis 3: Score 7/10 - Creative but may be computationally expensive"""
        
        elif system_prompt == "refiner":
            return """Refined approach: Implement a meta-learning framework using MAML (Model-Agnostic Meta-Learning) 
with task-specific adaptation layers. Use few-shot learning techniques to enable rapid adaptation 
to new tasks with minimal examples. Include a memory-augmented component for storing and retrieving 
successful strategies."""
        
        elif system_prompt == "summarizer":
            return "Progress: Goal decomposed into subgoals, multiple hypotheses generated and evaluated, top hypothesis refined."
        
        return "Mock LLM response"


# ============================================================================
# Main System Interface
# ============================================================================

class CognitiveSystem:
    """High-level interface for the cognitive architecture."""
    
    def __init__(self, llm=None, knowledge_base: Optional[List[str]] = None):
        """Initialize the cognitive system."""
        self.llm = llm or MockLLM()
        self.skill_core = SkillCore(self.llm)
        self.working_mem = WorkingMemory()
        self.episodic_mem = EpisodicMemory()
        self.semantic_mem = SemanticMemory(knowledge_base)
        self.controller = Controller(
            self.skill_core,
            self.working_mem,
            self.episodic_mem,
            self.semantic_mem
        )
    
    def solve(self, goal: str, max_steps: int = 20) -> Episode:
        """Solve a problem given a goal."""
        self.working_mem.clear()
        self.controller.set_goal(goal)
        episode = self.controller.run_until_done(max_steps)
        return episode
    
    def add_knowledge(self, knowledge: str):
        """Add knowledge to semantic memory."""
        self.semantic_mem.add_knowledge(knowledge)
    
    def get_episodes(self) -> List[Episode]:
        """Get all stored episodes."""
        return self.episodic_mem.get_all_episodes()
    
    def get_working_memory_state(self) -> Dict[str, Any]:
        """Get current working memory state."""
        return {
            "goal": self.working_mem.goal,
            "subgoals": self.working_mem.subgoals,
            "hypotheses": self.working_mem.hypotheses,
            "context_notes": self.working_mem.context_notes
        }
