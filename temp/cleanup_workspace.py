#!/usr/bin/env python3
"""
Cleanup Script - Remove Unnecessary Files
Keeps only the consolidated directory and essential documentation
"""

import os
import shutil
from pathlib import Path

def cleanup_workspace():
    """Remove unnecessary files and keep only essentials"""
    base_path = Path("c:/Users/ramkr/hello-world/temp")
    
    # Files and directories to keep
    keep_items = {
        "consolidated",  # Our consolidated system
        ".venv",        # Python virtual environment  
        ".copilotignore", # Copilot ignore file
        "README.md",    # Will be replaced with consolidated version
        "requirements.txt" # Will be replaced with consolidated version
    }
    
    print("🧹 Cleaning up workspace...")
    print("=" * 50)
    
    # Get all items in the base directory
    all_items = list(base_path.iterdir())
    
    removed_count = 0
    kept_count = 0
    
    for item in all_items:
        if item.name in keep_items:
            print(f"✅ Keeping: {item.name}")
            kept_count += 1
        else:
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"🗑️ Removed directory: {item.name}")
                else:
                    item.unlink()
                    print(f"🗑️ Removed file: {item.name}")
                removed_count += 1
            except Exception as e:
                print(f"⚠️ Could not remove {item.name}: {e}")
    
    print("=" * 50)
    print(f"📊 Cleanup Summary:")
    print(f"   • Kept: {kept_count} items")
    print(f"   • Removed: {removed_count} items")
    
    # Copy consolidated files to root level
    print("\n📋 Setting up final structure...")
    
    # Copy README from consolidated
    consolidated_readme = base_path / "consolidated" / "README.md"
    root_readme = base_path / "README.md"
    
    if consolidated_readme.exists():
        shutil.copy2(consolidated_readme, root_readme)
        print("✅ Updated root README.md")
    
    # Copy requirements from consolidated  
    consolidated_req = base_path / "consolidated" / "requirements.txt"
    root_req = base_path / "requirements.txt"
    
    if consolidated_req.exists():
        shutil.copy2(consolidated_req, root_req)
        print("✅ Updated root requirements.txt")
    
    print("\n🎯 Final Structure:")
    print("   temp/")
    print("   ├── README.md (comprehensive documentation)")
    print("   ├── requirements.txt")
    print("   ├── .venv/ (Python environment)")
    print("   └── consolidated/ (complete working system)")
    print("       ├── src/consortium/ (core services)")
    print("       ├── templates/ (web UI)")
    print("       ├── models/ (ML models)")  
    print("       ├── docs/ (architecture)")
    print("       └── *.py (launch scripts & demos)")
    
    print("\n✅ Workspace cleanup complete!")
    print("🚀 To run the system: cd consolidated && python launch_system.py")

if __name__ == "__main__":
    cleanup_workspace()
