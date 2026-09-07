import os
import glob

# Remove problematic emojis from all agent files to prevent Windows console crashes
emojis = {"📍": "[Location]", "🧬": "[Genetics]", "✅": "[Done]", "⚠️": "[Warning]", "🧠": "[Brain]", "🩺": "[Clinical]"}

for filepath in glob.glob('agents/**/*.py', recursive=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    for emoji, text in emojis.items():
        if emoji in content:
            content = content.replace(emoji, text)
            modified = True
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned emojis from {filepath}")
