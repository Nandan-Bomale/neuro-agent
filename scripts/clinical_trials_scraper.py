import argparse
import requests
import json
import sys

def get_clinical_trials(tumor_type: str, mutation_status: str, max_results: int = 3):
    """
    Scrapes ClinicalTrials.gov via their v2 API to find recruiting trials
    for a given tumor type and mutation status.
    """
    # ClinicalTrials.gov API v2 endpoint for studies
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # We want to search for the condition (tumor type) and any additional terms (mutation status)
    # We also want to filter by recruiting status.
    params = {
        "query.cond": tumor_type,
        "query.term": mutation_status,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
        # Specify what fields we want back to keep the response light
        "fields": "NCTId,Condition,BriefTitle,OverallStatus,Phase,EligibilityModule"
    }

    try:
        print(f"Fetching clinical trials for {tumor_type} with {mutation_status}...")
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        
        data = response.json()
        
        studies = data.get('studies', [])
        
        if not studies:
            print("No matching clinical trials found.")
            return []
            
        results = []
        for study in studies:
            protocol_section = study.get('protocolSection', {})
            
            # Extract basic info
            nct_id = protocol_section.get('identificationModule', {}).get('nctId', 'Unknown')
            brief_title = protocol_section.get('identificationModule', {}).get('briefTitle', 'No Title')
            
            # Phase info
            phases = protocol_section.get('designModule', {}).get('phases', [])
            phase_str = ", ".join(phases) if phases else "Unknown Phase"
            
            # Conditions
            conditions = protocol_section.get('conditionsModule', {}).get('conditions', [])
            condition_str = ", ".join(conditions) if conditions else "Unknown Condition"

            # Criteria 
            eligibility = protocol_section.get('eligibilityModule', {}).get('eligibilityCriteria', 'No criteria listed.')
            
            trial_info = {
                "NCTId": nct_id,
                "Title": brief_title,
                "Phase": phase_str,
                "Conditions": condition_str,
                "EligibilityCriteria": eligibility[:500] + "..." if len(eligibility) > 500 else eligibility,
                "URL": f"https://clinicaltrials.gov/study/{nct_id}"
            }
            results.append(trial_info)
            
        return results

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ClinicalTrials.gov API: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Find active clinical trials based on tumor type and mutation status.")
    parser.add_argument("--tumor", type=str, required=True, help="Tumor type (e.g., 'Glioblastoma')")
    parser.add_argument("--mutation", type=str, required=True, help="Mutation status (e.g., 'IDH-mutant')")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of trials to return (default: 3)")
    
    args = parser.parse_args()
    
    trials = get_clinical_trials(args.tumor, args.mutation, args.limit)
    
    if trials:
        print(f"\n--- Top {len(trials)} Recruiting Trials ---")
        for i, trial in enumerate(trials, 1):
            print(f"\n{i}. {trial['Title']}")
            print(f"   NCT ID: {trial['NCTId']}")
            print(f"   Phase: {trial['Phase']}")
            print(f"   Conditions: {trial['Conditions']}")
            print(f"   URL: {trial['URL']}")
            print(f"   Criteria Snippet: {trial['EligibilityCriteria']}")

if __name__ == "__main__":
    main()
