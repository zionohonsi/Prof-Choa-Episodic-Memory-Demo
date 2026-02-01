"""
Unit tests for the cognitive architecture system.
"""

import unittest
from cognitive_system import (
    WorkingMemory, Episode, EpisodicMemory, SemanticMemory,
    SkillCore, Controller, CognitiveSystem, MockLLM, ACTIONS
)


class TestWorkingMemory(unittest.TestCase):
    """Test WorkingMemory class."""
    
    def test_initialization(self):
        """Test working memory initialization."""
        wm = WorkingMemory()
        self.assertIsNone(wm.goal)
        self.assertEqual(wm.subgoals, [])
        self.assertEqual(wm.hypotheses, [])
        self.assertEqual(wm.intermediate_results, [])
        self.assertEqual(wm.context_notes, [])
    
    def test_clear(self):
        """Test clearing working memory."""
        wm = WorkingMemory()
        wm.goal = "Test goal"
        wm.subgoals = ["subgoal1"]
        wm.clear()
        self.assertIsNone(wm.goal)
        self.assertEqual(wm.subgoals, [])


class TestEpisode(unittest.TestCase):
    """Test Episode class."""
    
    def test_episode_creation(self):
        """Test episode creation."""
        episode = Episode(
            goal="Test goal",
            steps=[("ACTION1", "result1")],
            outcome="success",
            notes=["note1"]
        )
        self.assertEqual(episode.goal, "Test goal")
        self.assertEqual(len(episode.steps), 1)
        self.assertEqual(episode.outcome, "success")
        self.assertEqual(len(episode.notes), 1)
    
    def test_to_dict(self):
        """Test episode serialization."""
        episode = Episode(
            goal="Test goal",
            steps=[("ACTION1", "result1")],
            outcome="success"
        )
        episode_dict = episode.to_dict()
        self.assertIn("goal", episode_dict)
        self.assertIn("steps", episode_dict)
        self.assertIn("outcome", episode_dict)


class TestEpisodicMemory(unittest.TestCase):
    """Test EpisodicMemory class."""
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving episodes."""
        memory = EpisodicMemory()
        episode = Episode(
            goal="machine learning optimization",
            steps=[],
            outcome="success"
        )
        memory.store_episode(episode)
        
        # Retrieve similar episodes
        similar = memory.retrieve_similar("machine learning", k=1)
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0].goal, episode.goal)
    
    def test_empty_retrieval(self):
        """Test retrieval from empty memory."""
        memory = EpisodicMemory()
        similar = memory.retrieve_similar("test", k=3)
        self.assertEqual(similar, [])


class TestSemanticMemory(unittest.TestCase):
    """Test SemanticMemory class."""
    
    def test_add_and_retrieve_knowledge(self):
        """Test adding and retrieving knowledge."""
        memory = SemanticMemory()
        memory.add_knowledge("Neural networks use backpropagation")
        memory.add_knowledge("Deep learning requires large datasets")
        
        relevant = memory.retrieve_relevant("neural networks", k=1)
        self.assertEqual(len(relevant), 1)
        self.assertIn("Neural networks", relevant[0])
    
    def test_initialization_with_knowledge(self):
        """Test initialization with knowledge base."""
        knowledge = ["fact1", "fact2"]
        memory = SemanticMemory(knowledge)
        self.assertEqual(len(memory.knowledge_base), 2)


class TestSkillCore(unittest.TestCase):
    """Test SkillCore class."""
    
    def test_llm_call(self):
        """Test LLM call wrapper."""
        llm = MockLLM()
        skill_core = SkillCore(llm)
        result = skill_core.call("planner", "test prompt")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestController(unittest.TestCase):
    """Test Controller class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.skill_core = SkillCore(self.llm)
        self.working_mem = WorkingMemory()
        self.episodic_mem = EpisodicMemory()
        self.semantic_mem = SemanticMemory()
        self.controller = Controller(
            self.skill_core,
            self.working_mem,
            self.episodic_mem,
            self.semantic_mem
        )
    
    def test_set_goal(self):
        """Test setting goal."""
        self.controller.set_goal("Test goal")
        self.assertEqual(self.working_mem.goal, "Test goal")
        self.assertEqual(len(self.controller.trace), 0)
    
    def test_select_action(self):
        """Test action selection."""
        self.controller.set_goal("Test goal")
        action = self.controller.select_action()
        self.assertIn(action, ACTIONS)
        self.assertEqual(action, "DECOMPOSE_GOAL")  # First action should be decompose
    
    def test_execute_action_decompose(self):
        """Test executing decompose action."""
        self.controller.set_goal("Test goal")
        result = self.controller.execute_action("DECOMPOSE_GOAL")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_step(self):
        """Test single reasoning step."""
        self.controller.set_goal("Test goal")
        action, result = self.controller.step()
        self.assertIn(action, ACTIONS)
        self.assertEqual(len(self.controller.trace), 1)
    
    def test_goal_satisfied(self):
        """Test goal satisfaction check."""
        self.controller.set_goal("Test goal")
        self.assertFalse(self.controller.goal_satisfied())
        
        # Add hypotheses and results
        self.working_mem.hypotheses = ["hypothesis1"]
        self.working_mem.intermediate_results = ["result1"]
        self.assertTrue(self.controller.goal_satisfied())
    
    def test_run_until_done(self):
        """Test full cognitive loop."""
        self.controller.set_goal("Design a new algorithm")
        episode = self.controller.run_until_done(max_steps=10)
        
        self.assertIsInstance(episode, Episode)
        self.assertEqual(episode.goal, "Design a new algorithm")
        self.assertGreater(len(episode.steps), 0)
        self.assertIn(episode.outcome, ["success", "partial", "failure"])


class TestCognitiveSystem(unittest.TestCase):
    """Test CognitiveSystem class."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = CognitiveSystem()
        self.assertIsNotNone(system.skill_core)
        self.assertIsNotNone(system.working_mem)
        self.assertIsNotNone(system.episodic_mem)
        self.assertIsNotNone(system.semantic_mem)
        self.assertIsNotNone(system.controller)
    
    def test_add_knowledge(self):
        """Test adding knowledge."""
        system = CognitiveSystem()
        system.add_knowledge("Test knowledge")
        self.assertEqual(len(system.semantic_mem.knowledge_base), 1)
    
    def test_solve(self):
        """Test problem solving."""
        system = CognitiveSystem()
        episode = system.solve("Test goal", max_steps=5)
        
        self.assertIsInstance(episode, Episode)
        self.assertEqual(episode.goal, "Test goal")
        self.assertGreater(len(episode.steps), 0)
    
    def test_get_episodes(self):
        """Test retrieving episodes."""
        system = CognitiveSystem()
        system.solve("Goal 1", max_steps=3)
        system.solve("Goal 2", max_steps=3)
        
        episodes = system.get_episodes()
        self.assertEqual(len(episodes), 2)
    
    def test_get_working_memory_state(self):
        """Test getting working memory state."""
        system = CognitiveSystem()
        system.solve("Test goal", max_steps=3)
        
        state = system.get_working_memory_state()
        self.assertIn("goal", state)
        self.assertIn("subgoals", state)
        self.assertIn("hypotheses", state)
        self.assertIn("context_notes", state)


class TestParsingMethods(unittest.TestCase):
    """Test LLM output parsing methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.skill_core = SkillCore(self.llm)
        self.working_mem = WorkingMemory()
        self.episodic_mem = EpisodicMemory()
        self.semantic_mem = SemanticMemory()
        self.controller = Controller(
            self.skill_core,
            self.working_mem,
            self.episodic_mem,
            self.semantic_mem
        )
    
    def test_parse_subgoals(self):
        """Test parsing subgoals from text."""
        text = """1. First subgoal
2. Second subgoal
3. Third subgoal"""
        result = self.controller._parse_subgoals(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "First subgoal")
    
    def test_parse_hypotheses(self):
        """Test parsing hypotheses from text."""
        text = """- Hypothesis one
- Hypothesis two
* Hypothesis three"""
        result = self.controller._parse_hypotheses(text)
        self.assertGreater(len(result), 0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    print("Running Cognitive Architecture Unit Tests\n")
    print("="*70)
    run_tests()
