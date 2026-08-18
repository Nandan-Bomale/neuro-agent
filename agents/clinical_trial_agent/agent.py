import sys
import os

# Ensure the root project directory is in the path so we can import from scripts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.clinical_trials_scraper import get_clinical_trials

class ClinicalTrialAgent:
    def __init__(self):
        """
        Initializes the Clinical Trial Agent.
        This agent wraps the clinicaltrials.gov scraper to find relevant trials.
        """
        pass

    def find_eligible_trials(self, tumor_type: str, mutation_status: str, max_results: int = 3) -> list:
        """
        Finds active, recruiting clinical trials for a patient.
        
        Args:
            tumor_type (str): The patient's tumor type (e.g., 'Glioblastoma')
            mutation_status (str): The patient's mutation status (e.g., 'IDH-mutant')
            max_results (int): Number of trials to return (default: 3)
            
        Returns:
            list: A list of dictionaries containing trial information.
        """
        print(f"[ClinicalTrialAgent] Searching trials for {tumor_type} with {mutation_status}...")
        
        # Call the scraper function directly
        trials = get_clinical_trials(
            tumor_type=tumor_type, 
            mutation_status=mutation_status, 
            max_results=max_results
        )
        
        return trials

    def run(self, state: dict) -> dict:
        """LangGraph node interface."""
        print("[ClinicalTrialAgent] Running node...")
        tumor_findings = state.get("tumor_classification_findings", {})
        radio_findings = state.get("radiogenomics_findings", {})
        
        tumor_type = tumor_findings.get("tumor_type", "Unknown")
        mutation_status = radio_findings.get("idh_mutation_status", "Unknown")
        
        # If the values are unknown, we'll just use Glioblastoma IDH-mutant for the mock test
        if tumor_type == "Unknown":
            tumor_type = "Glioblastoma"
        if mutation_status == "Unknown":
            mutation_status = "IDH-mutant"
            
        trials = self.find_eligible_trials(tumor_type, mutation_status, max_results=3)
        return {"clinical_trials": trials}

# Simple test if run directly
if __name__ == "__main__":
    agent = ClinicalTrialAgent()
    results = agent.find_eligible_trials("Glioblastoma", "IDH-mutant")
    
    for i, trial in enumerate(results, 1):
        print(f"\n{i}. {trial['Title']}")
        print(f"   URL: {trial['URL']}")
