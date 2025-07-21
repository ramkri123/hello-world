"""
Example: Adding context from files in different directories
"""

import sys
import os
import json
from pathlib import Path

# Method 1: Add directory to Python path for imports
external_module_path = r"C:\path\to\external\modules"
if external_module_path not in sys.path:
    sys.path.append(external_module_path)

# Now you can import modules from that directory
# import external_module

# Method 2: Read configuration from different directory
def load_external_config(config_path):
    """Load configuration from external directory"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in config file: {config_path}")
        return {}

# Method 3: Use pathlib for cross-platform path handling
def get_external_files(base_directory):
    """Get files from external directory"""
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"Directory does not exist: {base_directory}")
        return []
    
    return list(base_path.glob("*.json"))  # or any pattern

# Method 4: Environment variables for dynamic paths
def load_from_env_path():
    """Load from path specified in environment variable"""
    external_path = os.getenv('CONSORTIUM_EXTERNAL_PATH', 'default/path')
    config_file = os.path.join(external_path, 'config.json')
    
    if os.path.exists(config_file):
        return load_external_config(config_file)
    return {}

# Example usage in consortium context
if __name__ == "__main__":
    # Load external bank configurations
    external_config = load_external_config(r"C:\external\bank_configs\consortium_config.json")
    
    # Load external fraud rules
    fraud_rules_dir = r"C:\external\fraud_rules"
    fraud_rule_files = get_external_files(fraud_rules_dir)
    
    print(f"Loaded {len(fraud_rule_files)} fraud rule files")
