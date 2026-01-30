import os
from core.orchestrator import Orchestrator

def main():
    # Initialize System
    app = Orchestrator()

    print("QDoctor System Ready (Type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            response = app.process_query(user_input)
            print(f"QDoctor: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"System Error: {e}")

if __name__ == "__main__":
    main()