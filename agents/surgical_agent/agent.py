from typing import Any, Dict

class SurgicalAgent:
    def __init__(self):
        pass

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("[SurgicalAgent] Running mock assessment...")
        return {
            "surgical_analysis": {
                "resectability_score": 0.85,
                "eloquent_area_proximity": "low"
            }
        }
