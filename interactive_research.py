"""
Interactive Neuro-AI Research Tool
Allows you to solve your own AI/medical/brain problems interactively.
"""

import os
import sys
sys.path.insert(0, '/home/ubuntu/cognitive_architecture')

from neuro_ai_research_system import NeuroAIResearchSystem


def print_header():
    """Print welcome header."""
    print("\n" + "="*70)
    print(" 🧠 INTERACTIVE NEURO-AI RESEARCH SYSTEM")
    print("="*70)
    print("\nThis system can help you explore problems in:")
    print("  • Artificial Intelligence & Machine Learning")
    print("  • Neuroscience & Brain Function")
    print("  • Medical Applications of AI")
    print("  • Brain-Computer Interfaces")
    print("  • Neuromorphic Computing")
    print("  • Cognitive Science")
    print("\nThe system uses episodic memory - it learns from each problem!")
    print("="*70 + "\n")


def get_user_problem():
    """Get research problem from user."""
    print("\n" + "-"*70)
    print("Enter your research question or problem:")
    print("(Type 'examples' to see example questions)")
    print("(Type 'quit' to exit)")
    print("-"*70)
    
    user_input = input("\n> ").strip()
    
    if user_input.lower() == 'examples':
        show_examples()
        return get_user_problem()
    
    return user_input


def show_examples():
    """Show example research questions."""
    print("\n" + "="*70)
    print(" EXAMPLE RESEARCH QUESTIONS")
    print("="*70)
    
    examples = {
        "AI & Machine Learning": [
            "How can we reduce hallucinations in large language models?",
            "Design a more efficient attention mechanism for transformers",
            "What are promising approaches to few-shot learning?",
            "How can we improve interpretability of deep neural networks?"
        ],
        "Neuroscience & Brain": [
            "How does the brain process visual information hierarchically?",
            "What mechanisms enable rapid learning in the hippocampus?",
            "How do neural oscillations contribute to cognition?",
            "What is the role of glial cells in information processing?"
        ],
        "Brain-Inspired AI": [
            "How can spiking neural networks improve AI efficiency?",
            "Design an AI system inspired by the brain's predictive coding",
            "How can we implement working memory in neural networks?",
            "Create a continual learning system based on neuroplasticity"
        ],
        "Medical Applications": [
            "How can AI improve early detection of Alzheimer's disease?",
            "Design a brain-computer interface for paralysis patients",
            "What AI approaches can help in personalized medicine?",
            "How can deep learning assist in medical image diagnosis?"
        ]
    }
    
    for category, questions in examples.items():
        print(f"\n{category}:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
    
    print("\n" + "="*70)


def display_memory_stats(system):
    """Display current episodic memory statistics."""
    episodes = system.get_episodes()
    if not episodes:
        print("\n📚 Episodic Memory: Empty (no problems solved yet)")
        return
    
    print("\n" + "="*70)
    print(" 📚 EPISODIC MEMORY STATISTICS")
    print("="*70)
    print(f"Total episodes: {len(episodes)}")
    print(f"Successful: {sum(1 for e in episodes if e.outcome == 'success')}")
    print(f"Partial: {sum(1 for e in episodes if e.outcome == 'partial')}")
    
    print("\nRecent problems solved:")
    for i, ep in enumerate(episodes[-5:], 1):
        print(f"  {i}. {ep.goal[:60]}... ({ep.outcome})")
    
    print("="*70 + "\n")


def main():
    """Main interactive loop."""
    print_header()
    
    # Check for Gemini API key
    use_real_llm = False
    if os.getenv("GEMINI_API_KEY"):
        print("✓ Gemini API key found. Using Gemini for research.\n")
        use_real_llm = True
    else:
        print("⚠️  No GEMINI_API_KEY found.")
        print("   Set it with: export GEMINI_API_KEY='your-key-here'")
        print("   Using MockLLM for demonstration.\n")
    
    # Initialize system
    print("Initializing research system...")
    system = NeuroAIResearchSystem(use_real_llm=use_real_llm)
    print("✓ System ready!\n")
    
    # Optional: Load existing problems to build initial memory
    print("Would you like to build initial episodic memory with foundational problems?")
    build_initial = input("This helps the system perform better on advanced questions (y/n): ").lower().strip() == 'y'
    
    if build_initial:
        print("\nBuilding initial episodic memory...")
        initial_problems = [
            "How does attention mechanism in transformers relate to biological attention?",
            "What are key differences between artificial and biological neural networks?",
            "How can neuroplasticity principles improve continual learning in AI?"
        ]
        system.build_episodic_library(initial_problems, max_steps=10)
    
    # Main interaction loop
    problem_count = 0
    while True:
        # Show memory stats
        if problem_count > 0:
            display_memory_stats(system)
        
        # Get user problem
        problem = get_user_problem()
        
        if problem.lower() in ['quit', 'exit', 'q']:
            break
        
        if not problem:
            print("Please enter a valid research question.")
            continue
        
        # Solve the problem
        try:
            episode = system.solve_research_problem(problem, max_steps=15)
            problem_count += 1
            
            # Ask if user wants to continue
            print("\nOptions:")
            print("  1. Ask another question")
            print("  2. View episodic memory stats")
            print("  3. Save memory library")
            print("  4. Quit")
            
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == '2':
                display_memory_stats(system)
            elif choice == '3':
                filename = f"my_research_memory_{problem_count}_episodes.json"
                system.save_memory_library(filename)
                print(f"\n✓ Memory library saved to: {filename}")
            elif choice == '4':
                break
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")
    
    # Final summary
    print("\n" + "="*70)
    print(" SESSION SUMMARY")
    print("="*70)
    
    summary = system.get_research_summary()
    print(f"\nProblems solved this session: {problem_count}")
    print(f"Total episodes in memory: {summary['episodic_memory_size']}")
    
    if problem_count > 0:
        save = input("\nSave episodic memory library? (y/n): ").lower().strip() == 'y'
        if save:
            filename = f"research_memory_final_{summary['episodic_memory_size']}_episodes.json"
            system.save_memory_library(filename)
    
    print("\n" + "="*70)
    print(" Thank you for using the Neuro-AI Research System!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
