import sys
import os
from dotenv import load_dotenv  
load_dotenv()
# Ensure Python can find your 'src' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.planning_agent import PlanningAgent

def main():
    # 1. Make Object of the PlanningAgent class
    planner = PlanningAgent()

    # 2. Define your query (What you want the agent to find)
    # You can change this string to whatever you want, e.g., "curved monitors"
    query = "best laptop deals"

    print(f"\n🚀 Launching Agent with query: '{query}'...")

    # 3. Call the function
    planner.start_workflow(user_query=query)

if __name__ == "__main__":
    main()