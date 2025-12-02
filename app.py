import sys
import os
from dotenv import load_dotenv  
load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.planning_agent import PlanningAgent

def main():
    # 1. Make Object of the PlanningAgent class
    planner = PlanningAgent()

    # 2. Define your query (What you want the agent to find)
    query = "best laptop deals"

    print(f"\nLaunching Agent with query: '{query}'...")

    # 3. Call the function
    planner.start_workflow(user_query=query)

if __name__ == "__main__":
    main()
