#!/usr/bin/env python3
"""
Start Flask UI in dedicated terminal
"""
import sys
import os
import time
import requests

# Add paths for imports
sys.path.append(os.path.dirname(__file__))

def wait_for_consortium():
    """Wait for consortium to be available"""
    print("⏳ Waiting for consortium hub to be available...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            if response.status_code == 200:
                print("✅ Consortium hub is ready!")
                return True
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}/30)")
    
    print("⚠️  Consortium hub not available - UI will still start")
    return False

def main():
    print("🌐 FLASK UI - DEDICATED TERMINAL")
    print("=" * 50)
    print("🚀 Starting Flask Web Interface...")
    print("📍 UI URL: http://localhost:5000")
    print("🔗 Consortium API: http://localhost:8080")
    print("📊 Features:")
    print("   • 13 Fraud Scenarios (CEO BEC, Crypto, Romance, etc.)")
    print("   • Real-time Distributed Analysis")
    print("   • Privacy-Preserving Account Anonymization")
    print("   • Scenario-Aware Confidence Weighting")
    print("-" * 50)
    
    # Wait for consortium (optional)
    wait_for_consortium()
    
    print("🌐 Starting Flask UI...")
    
    try:
        # Import and run Flask app
        from flask_ui import app
        
        print("✅ Flask UI ready!")
        print("🌐 Open http://localhost:5000 in your browser")
        print("💡 Press Ctrl+C to stop the UI")
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Flask UI stopped by user")
    except Exception as e:
        print(f"❌ UI error: {e}")
        print("💡 Make sure flask_ui.py exists and is properly configured")

if __name__ == "__main__":
    main()
