#!/usr/bin/env python3
"""
Start Consortium Hub in dedicated terminal
"""
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.consortium.consortium_hub import ConsortiumHub

def main():
    print("🌟 CONSORTIUM HUB - DEDICATED TERMINAL")
    print("=" * 50)
    print("🚀 Starting Consortium Hub...")
    print("📍 Hub URL: http://localhost:8080")
    print("🔍 Health Check: http://localhost:8080/health")
    print("📊 API Endpoints:")
    print("   • POST /inference - Submit fraud detection requests")
    print("   • GET /results/<session_id> - Get analysis results")
    print("   • POST /register - Register bank participants")
    print("   • GET /poll_inference - Bank polling endpoint")
    print("-" * 50)
    
    try:
        hub = ConsortiumHub(port=8080)
        hub.run(debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Consortium Hub stopped by user")
    except Exception as e:
        print(f"❌ Hub error: {e}")

if __name__ == "__main__":
    main()
