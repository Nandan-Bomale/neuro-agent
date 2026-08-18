from typing import Any, Dict

class PrognosticAgent:
    def __init__(self):
        pass

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("[PrognosticAgent] Running mock survival prediction...")
        return {
            "prognostic_analysis": {
                "overall_survival_months": 18.5,
                "progression_free_survival_months": 12.0
            }
        }
