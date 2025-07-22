#!/usr/bin/env python3
"""
Stop all running services
"""
import subprocess
import psutil
import requests
import time

def stop_services_on_ports():
    """Stop services running on our ports"""
    ports = [8080, 5000]  # Consortium hub and Flask UI
    
    for port in ports:
        print(f"🔍 Checking port {port}...")
        
        # Find processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port:
                        print(f"   🛑 Stopping process {proc.info['pid']} ({proc.info['name']})")
                        proc.terminate()
                        time.sleep(1)
                        if proc.is_running():
                            proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

def main():
    print("🛑 STOPPING ALL CONSORTIUM SERVICES")
    print("=" * 40)
    
    print("🔍 Stopping services on known ports...")
    stop_services_on_ports()
    
    print("⏳ Waiting for cleanup...")
    time.sleep(3)
    
    # Check if services are stopped
    services_stopped = True
    
    for port, name in [(8080, "Consortium Hub"), (5000, "Flask UI")]:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"⚠️  {name} still running on port {port}")
                services_stopped = False
        except:
            print(f"✅ {name} stopped")
    
    if services_stopped:
        print("🎉 All services stopped successfully!")
    else:
        print("⚠️  Some services may still be running")
    
    print("💡 You can now restart with the updated scripts")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 You may need to manually close terminal windows")
