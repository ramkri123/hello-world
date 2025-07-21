#!/usr/bin/env python3
"""
Quick Launcher - Organized Project Structure
Simple commands for the reorganized consortium fraud detection system
"""

import subprocess
import sys
import os

def show_help():
    """Show available commands"""
    print("🚀 PRIVACY-PRESERVING CONSORTIUM - QUICK LAUNCHER")
    print("=" * 55)
    print()
    print("📋 Available Commands:")
    print()
    print("🌐 SYSTEM MANAGEMENT:")
    print("  start-all         Start complete distributed system")
    print("  start-hub         Start consortium hub only")
    print("  start-banks       Start all banks")
    print("  start-ui          Start web interface")
    print()
    print("🏦 INDIVIDUAL BANKS:")
    print("  start-bank-a      Start Bank A (Wire Transfer Specialist)")
    print("  start-bank-b      Start Bank B (Identity Verification)")
    print("  start-bank-c      Start Bank C (Network Pattern Analyst)")
    print("  list-banks        List available bank configurations")
    print()
    print("🧪 TESTING:")
    print("  test-privacy      Test privacy-preserving system")
    print("  test-distributed  Test distributed system")
    print("  test-client       Run simple client test")
    print()
    print("📚 UTILITIES:")
    print("  help              Show this help")
    print("  structure         Show project structure")
    print()
    print("💡 Examples:")
    print("  python launcher.py start-all")
    print("  python launcher.py start-bank-a")
    print("  python launcher.py test-privacy")

def run_command(script_path, *args):
    """Run a script with arguments"""
    try:
        cmd = [sys.executable, script_path] + list(args)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ File not found: {script_path}")
        sys.exit(1)

def main():
    """Main launcher"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    # System management commands
    if command == "start-all":
        run_command("scripts/start_unified_consortium.py")
    elif command == "start-hub":
        run_command("src/consortium/consortium_hub.py")
    elif command == "start-banks":
        run_command("scripts/start_banks_separately.py")
    elif command == "start-ui":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/consortium_fraud_ui.py"])
    
    # Individual bank commands
    elif command == "start-bank-a":
        run_command("src/banks/universal_bank_launcher.py", "bank_A")
    elif command == "start-bank-b":
        run_command("src/banks/universal_bank_launcher.py", "bank_B")
    elif command == "start-bank-c":
        run_command("src/banks/universal_bank_launcher.py", "bank_C")
    elif command == "list-banks":
        run_command("src/banks/universal_bank_launcher.py", "--list-banks")
    
    # Testing commands
    elif command == "test-privacy":
        run_command("tests/test_privacy_consortium.py")
    elif command == "test-distributed":
        run_command("tests/test_distributed_consortium.py")
    elif command == "test-client":
        run_command("src/consortium/consortium_client.py")
    
    # Utility commands
    elif command == "help":
        show_help()
    elif command == "structure":
        with open("docs/PROJECT_STRUCTURE.md", "r") as f:
            print(f.read())
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python launcher.py help' for available commands")
        sys.exit(1)

if __name__ == "__main__":
    main()
