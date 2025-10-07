import sys, time
from pathlib import Path

# Adjust sys.path to ensure imports work when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in map(str, sys.path):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "app"))
from app import agent_runner


QUERY = "What are the side effects of Ozempic?"


if __name__ == "__main__":
    workflow = agent_runner.AgentWorkflow()
    result = workflow.run(QUERY)
    print(result)
