"""
deploy_to_huggingface.py
-------------------------
Automated 1-Click Deployment of NeuroAgent to Hugging Face Spaces.
Provides 24/7 cloud hosting with 16 GB RAM, 2 vCPUs, 100% free,
independent of your laptop.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, get_token, login

def main():
    print("=" * 75)
    print("   NEUROAGENT - 24/7 FREE INDEPENDENT CLOUD DEPLOYMENT")
    print("=" * 75)
    print("\nThis script deploys NeuroAgent to Hugging Face Spaces Docker Cloud.")
    print("Benefits:")
    print("  * 100% Free Forever ($0.00, NO credit card needed)")
    print("  * 24/7 Uptime (Stays online even when your laptop is turned OFF)")
    print("  * 16 GB RAM + 2 vCPUs dedicated in the cloud")
    print("  * Generates a permanent public link anyone can access\n")

    api = HfApi()
    current_token = get_token()
    token = None

    # Check existing token role if available
    needs_new_token = True
    if current_token:
        try:
            info = api.whoami(token=current_token)
            username = info.get("name", "Unknown")
            role = info.get("auth", {}).get("accessToken", {}).get("role", "")
            print(f"[*] Found saved account: '{username}'")
            if role == "read":
                print(f"[!] Current token is READ-ONLY (role: 'read').")
                print("    Creating and uploading a Space requires a WRITE token.")
                needs_new_token = True
            elif role == "write":
                print(f"[+] Token has WRITE permissions!")
                choice = input("Do you want to use this saved token? (Y/n): ").strip().lower()
                if choice in ("", "y", "yes"):
                    token = current_token
                    needs_new_token = False
        except Exception:
            needs_new_token = True

    if needs_new_token:
        print("\n" + "-" * 75)
        print("To deploy 24/7 freely, get a WRITE token:")
        print("1. Open: https://huggingface.co/settings/tokens/new?tokenType=write")
        print("2. Set Token name: deploy")
        print("3. Ensure Token type is: WRITE (not Read)")
        print("4. Click 'Create token' and copy the 'hf_...' key")
        print("-" * 75)
        
        while True:
            user_input = input("\nPaste your Hugging Face WRITE Token here: ").strip()
            if not user_input:
                print("Deployment cancelled.")
                return
            
            try:
                info = api.whoami(token=user_input)
                username = info.get("name")
                role = info.get("auth", {}).get("accessToken", {}).get("role", "")
                if role == "read":
                    print(f"\n[-] That token is still READ-ONLY for user '{username}'.")
                    print("    Please create a token with 'Write' permission at:")
                    print("    https://huggingface.co/settings/tokens/new?tokenType=write")
                    continue
                
                token = user_input
                login(token=token)
                print(f"\n[+] Verified! Logged in as: {username} (Role: {role or 'write'})")
                break
            except Exception as e:
                print(f"[-] Token authentication failed: {e}")
                retry = input("Try again? (Y/n): ").strip().lower()
                if retry in ("n", "no"):
                    return

    user_info = api.whoami(token=token)
    username = user_info["name"]
    space_name = "neuro-agent"
    repo_id = f"{username}/{space_name}"
    print(f"\n[+] Target Space URL: https://huggingface.co/spaces/{repo_id}")

    print("\n[1/3] Creating Space on Hugging Face (Docker SDK)...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=token
        )
        print("    Space repository verified!")
    except Exception as e:
        print(f"[-] Error creating space: {e}")
        return

    ignore_patterns = [
        "*.pyc",
        "__pycache__/**",
        "venv/**",
        ".venv/**",
        "env/**",
        "node_modules/**",
        "frontend_react/node_modules/**",
        "data/raw/**",
        "data/tumor_classification/**",
        "*.zip",
        "*.log",
        "runs/**",
        "backups/**",
        "scratch*/**",
        "scratch*.*",
        "models/tumor_classifier/type_ensemble_ep*.pth",
        "models/tumor_classifier/*backup*.pth",
        "models/vision/latest_checkpoint.pth",
        "models/vision/final_model.pth",
        "models/vision/best_model.pth",
        "models/vision/test_outputs/**",
        "tools/**",
        ".git/**"
    ]

    print("\n[2/3] Synchronizing production models and files (~980 MB)...")
    print("    (Uploading backend, 11 agents, compiled React UI, and trained AI models)")
    print("    Please wait while files upload to Hugging Face...")

    workspace_dir = Path(__file__).resolve().parent
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(workspace_dir),
            ignore_patterns=ignore_patterns,
            commit_message="Deploy production NeuroAgent multi-agent diagnostic system",
            token=token
        )
        print("\n[3/3] Upload complete!")
    except Exception as e:
        print(f"\n[-] Error during upload: {e}")
        return

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n" + "=" * 75)
    print("  SUCCESSFULLY DEPLOYED TO 24/7 CLOUD!")
    print("=" * 75)
    print(f"\nYour application is now building in the cloud at:\n  --> {space_url}\n")
    print("Hugging Face will automatically build the Docker container in ~3-5 minutes.")
    print("Once built, the app is permanently online 24/7, completely independent")
    print("of your laptop! Anyone can open that link from any browser.")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
