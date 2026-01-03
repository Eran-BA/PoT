#!/usr/bin/env python3
"""Upload Sudoku demo to HuggingFace Spaces."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import HfApi, upload_folder

SPACE_DIR = Path(__file__).parent.parent / "spaces" / "sudoku_demo"
SPACE_ID = "Eran92/pot-sudoku-solver"  # Change to your HF username

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in .env file")
        sys.exit(1)
    
    api = HfApi(token=token)
    
    # Create Space if it doesn't exist
    try:
        api.create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
        print(f"Created/verified Space: {SPACE_ID}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload files
    print(f"Uploading files from {SPACE_DIR}...")
    
    upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
        token=token,
    )
    
    print(f"\n✅ Space uploaded!")
    print(f"🔗 https://huggingface.co/spaces/{SPACE_ID}")

if __name__ == "__main__":
    main()

