import os
import sys
from src import Agent

def main():
    print("=" * 60)
    print(" Ultimate Local AI Agent (Files + ML + Web + Training Log) ")
    print(" Type 'exit' or 'quit' to close and save training data.")
    print("=" * 60 + "\n")

    chat_history = []

    training_data_path = os.path.join("data", "processed", "agent_training_data.jsonl")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                if chat_history:
                    print("\nSaving training data...")
                    print(Agent.save_training_data(chat_history))
                    print(f"Data saved to {training_data_path}")
                print("\nGoodbye!")
                break
            if not user_input:
                continue

            print("\nAgent is working...")
            final_answer, chat_history = Agent.Agent_run_arc(user_input, chat_history)
            print(f"\nAgent:\n{final_answer}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            if chat_history:
                print("\nSaving training data...")
                print(Agent.save_training_data(chat_history))
                print(f"Data saved to {training_data_path}")           
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")


if __name__ == "__main__":
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("agent_output", exist_ok=True)
    main()