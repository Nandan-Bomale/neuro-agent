import subprocess
import sys

print("Running radiogenomics...")
try:
    result = subprocess.run(
        [sys.executable, "-m", "agents.radiogenomics_agent.train", 
         "--csv-path", "data/radiogenomics_labels.csv", 
         "--data-dir", "data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData", 
         "--batch-size", "1", 
         "--num-workers", "2", 
         "--epochs", "2"],
        capture_output=True, text=True
    )
    print("RETURN CODE:", result.returncode)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
except Exception as e:
    print("EXCEPTION:", e)

print("Running neuro_oncologist...")
try:
    result = subprocess.run(
        [sys.executable, "-m", "agents.neuro_oncologist_agent.train"],
        capture_output=True, text=True
    )
    print("RETURN CODE:", result.returncode)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
except Exception as e:
    print("EXCEPTION:", e)
