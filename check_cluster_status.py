#!/usr/bin/env python3
"""Check status of RunPod cluster and get SSH commands."""

import json
import requests
import argparse
import time

def check_cluster_status(api_key):
    """Check status of all pods and get SSH info."""
    
    # Load cluster info
    try:
        with open("cluster_info_safe.json", "r") as f:
            cluster_info = json.load(f)
    except FileNotFoundError:
        print("❌ No cluster info found. Run deployment first.")
        return
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    query = """
    query GetPods {
        myself {
            pods {
                id
                name
                desiredStatus
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
            }
        }
    }
    """
    
    response = requests.post(
        "https://api.runpod.io/graphql",
        headers=headers,
        json={"query": query}
    )
    
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    result = response.json()
    pods = result["data"]["myself"]["pods"]
    
    # Match pods with cluster info
    cluster_pods = {node["id"]: node for node in cluster_info["nodes"]}
    
    print("\n🖥️  CLUSTER STATUS")
    print("=" * 60)
    
    master_ip = None
    all_ready = True
    
    for pod in pods:
        if pod["id"] in cluster_pods:
            name = cluster_pods[pod["id"]]["name"]
            status = pod.get("desiredStatus", "UNKNOWN")
            runtime = pod.get("runtime", {})
            uptime = runtime.get("uptimeInSeconds", 0) if runtime else 0
            
            print(f"\n📦 {name}")
            print(f"   Status: {status}")
            print(f"   Uptime: {uptime}s")
            
            if status == "RUNNING" and runtime and runtime.get("ports"):
                # Get SSH info
                ssh_found = False
                for port in runtime["ports"]:
                    if port.get("isIpPublic") and port.get("publicPort"):
                        print(f"   Port: {port.get('type', 'unknown')} - {port['ip']}:{port['publicPort']}")
                        if port.get("publicPort") == 22 or port.get("privatePort") == 22:
                            print(f"   SSH: ssh root@{port['ip']} -p {port['publicPort']}")
                            ssh_found = True
                if not ssh_found:
                    # Try to find any SSH-like port
                    for port in runtime["ports"]:
                        if port.get("isIpPublic") and port.get("publicPort") and port.get("publicPort") > 20000:
                            print(f"   SSH (guessed): ssh root@{port['ip']} -p {port['publicPort']}")
                    
                # Get internal IP for master
                if name.endswith("-0-autoclean") and not master_ip:
                    for port in runtime["ports"]:
                        if not port.get("isIpPublic"):
                            master_ip = port["ip"]
                            break
            else:
                all_ready = False
                print("   SSH: Not ready yet...")
    
    print("\n" + "=" * 60)
    
    if all_ready:
        print("✅ All nodes are ready!")
        print(f"\n🌐 Master IP: {master_ip or 'Could not determine'}")
        print("\n📝 Next steps:")
        print("1. SSH into each node")
        print("2. Update MASTER_ADDR in monitored_train.sh if needed")
        print("3. Run: bash < monitored_train.sh")
    else:
        print("⏳ Some nodes are still starting up...")
        print("Run this script again in a minute to check status.")
    
    # Show cost info
    elapsed_hours = (time.time() - time.mktime(time.strptime(cluster_info["created_at"][:19], "%Y-%m-%dT%H:%M:%S"))) / 3600
    total_cost = elapsed_hours * cluster_info["cost_per_hour"]
    print(f"\n💰 Current cost: ${total_cost:.2f} (${cluster_info['cost_per_hour']:.2f}/hr)")
    print(f"⏰ Time remaining: {cluster_info['max_hours'] - elapsed_hours:.1f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    args = parser.parse_args()
    
    check_cluster_status(args.api_key)