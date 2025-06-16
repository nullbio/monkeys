#!/usr/bin/env python3
"""
Safe RunPod cluster deployment with auto-cleanup and error handling.
Automatically terminates cluster on completion or error.
"""

import os
import json
import time
import subprocess
import requests
import signal
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import argparse


class SafeRunPodDeployer:
    """Deploy with automatic cleanup and cost protection."""
    
    def __init__(self, api_key: str, max_hours: float = 6.0, max_cost: float = 20.0, gpu_type: str = "NVIDIA RTX A4000"):
        self.api_key = api_key
        self.base_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.max_hours = max_hours
        self.max_cost = max_cost
        self.gpu_type = gpu_type
        self.start_time = time.time()
        self.node_ids = []
        self.cost_per_hour = 0
        self.cleanup_done = False
        
        # Register cleanup handlers
        signal.signal(signal.SIGINT, self._emergency_cleanup)
        signal.signal(signal.SIGTERM, self._emergency_cleanup)
        
    def _emergency_cleanup(self, signum, frame):
        """Emergency cleanup on interrupt."""
        print("\n⚠️  INTERRUPT DETECTED - CLEANING UP CLUSTER...")
        self.cleanup_cluster(self.node_ids)
        sys.exit(1)
        
    def create_cluster_with_timeout(self, num_nodes: int = 5):
        """Create cluster with automatic timeout."""
        print(f"Creating cluster with {num_nodes} nodes...")
        print(f"⏰ Auto-cleanup after {self.max_hours} hours or ${self.max_cost}")
        
        nodes = []
        
        # Create each node with error handling
        for i in range(num_nodes):
            try:
                node = self._create_single_node(i)
                if node:
                    nodes.append(node)
                    self.node_ids.append(node['id'])
                    self.cost_per_hour += float(node['costPerHr'])
                else:
                    print(f"❌ Failed to create node {i}")
                    
            except Exception as e:
                print(f"❌ Error creating node {i}: {e}")
                # Cleanup any created nodes
                if self.node_ids:
                    print("Cleaning up partial cluster...")
                    self.cleanup_cluster(self.node_ids)
                raise
                
        print(f"✅ Cluster created - Total cost: ${self.cost_per_hour:.2f}/hour")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_costs, daemon=True)
        monitor_thread.start()
        
        return nodes
    
    def _create_single_node(self, node_index: int):
        """Create a single node with RunPod API."""
        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                machineId
                costPerHr
                runtime {
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
        """
        
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuCount": 1,
                "volumeInGb": 10,
                "containerDiskInGb": 10,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
                "gpuTypeId": self.gpu_type,
                "name": f"agent-{node_index}-autoclean",
                "dockerArgs": "",
                "volumeMountPath": "/workspace",
                "env": [
                    {"key": "JUPYTER_PASSWORD", "value": "agent"},
                    {"key": "NODE_RANK", "value": str(node_index)},
                    {"key": "AUTO_CLEANUP_HOURS", "value": str(self.max_hours)}
                ],
                "imageName": "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04"
            }
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json={"query": mutation, "variables": variables},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result and result["data"]["podFindAndDeployOnDemand"]:
                return result["data"]["podFindAndDeployOnDemand"]
            else:
                # Print detailed error information
                print(f"Node {node_index} creation failed. Response: {json.dumps(result, indent=2)}")
                if "errors" in result:
                    for error in result["errors"]:
                        print(f"  Error: {error.get('message', 'Unknown error')}")
        else:
            print(f"Node {node_index} HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
        
        return None
    
    def _monitor_costs(self):
        """Monitor costs and time, auto-cleanup when limits reached."""
        while not self.cleanup_done:
            elapsed_hours = (time.time() - self.start_time) / 3600
            total_cost = elapsed_hours * self.cost_per_hour
            
            # Check limits
            if elapsed_hours >= self.max_hours:
                print(f"\n⏰ TIME LIMIT REACHED ({self.max_hours} hours) - AUTO CLEANUP")
                self.cleanup_cluster(self.node_ids)
                break
                
            if total_cost >= self.max_cost:
                print(f"\n💰 COST LIMIT REACHED (${self.max_cost}) - AUTO CLEANUP")
                self.cleanup_cluster(self.node_ids)
                break
                
            # Check for errors/completion
            if self._check_training_status() in ["error", "completed"]:
                print(f"\n🏁 Training {self._check_training_status()} - AUTO CLEANUP")
                self.cleanup_cluster(self.node_ids)
                break
                
            time.sleep(60)  # Check every minute
            
    def _check_training_status(self):
        """Check if training is still running, completed, or errored."""
        # Check pod status via API
        query = """
        query GetPods($input: PodFilter) {
            myself {
                pods(input: $input) {
                    id
                    name
                    runtime {
                        status
                        uptimeInSeconds
                    }
                }
            }
        }
        """
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={"query": query, "variables": {"input": {}}},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                pods = result["data"]["myself"]["pods"]
                
                # Check if any pods have stopped
                for pod in pods:
                    if pod["id"] in self.node_ids and pod["runtime"]["status"] != "RUNNING":
                        return "error"
                        
            # TODO: Check actual training logs/checkpoints
            # This would require SSH access to nodes
            
        except:
            pass
            
        return "running"
    
    def deploy_with_monitoring(self, nodes: list):
        """Deploy training script with error handling and monitoring."""
        
        # Generate monitored training script
        training_script = self._generate_monitored_script(nodes)
        
        with open("monitored_train.sh", "w") as f:
            f.write(training_script)
            
        print("\n📋 Deployment Instructions:")
        print("="*50)
        
        for i, node in enumerate(nodes):
            print(f"\nNode {i}:")
            # Find SSH info
            ssh_port = 22
            ssh_ip = "localhost"
            runtime = node.get("runtime")
            if runtime and runtime.get("ports"):
                for port in runtime.get("ports", []):
                    if port["type"] == "ssh" and port["isIpPublic"]:
                        ssh_ip = port["ip"]
                        ssh_port = port["publicPort"]
                    
            print(f"ssh root@{ssh_ip} -p {ssh_port}")
            print("Then run: bash < monitored_train.sh")
            
        return training_script
    
    def _generate_monitored_script(self, nodes: list):
        """Generate training script with error handling."""
        
        master_ip = self._get_master_ip(nodes)
        
        return f"""#!/bin/bash
# Auto-generated monitored training script
set -e  # Exit on error

# Configuration
export MASTER_ADDR={master_ip}
export MASTER_PORT=29500
export WORLD_SIZE={len(nodes)}
export MAX_HOURS={int(self.max_hours)}
export START_TIME=$(date +%s)

# Error handler
error_handler() {{
    echo "❌ ERROR DETECTED - Training failed!"
    echo "Error at line $1"
    # Signal completion to monitoring
    touch /workspace/training_failed
    exit 1
}}
trap 'error_handler $LINENO' ERR

# Time limit handler
check_time_limit() {{
    while true; do
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        if [ $ELAPSED -gt $((MAX_HOURS * 3600)) ]; then
            echo "⏰ Time limit reached - stopping training"
            pkill -f train_distributed.py
            touch /workspace/training_timeout
            exit 0
        fi
        sleep 300  # Check every 5 minutes
    done
}}
check_time_limit &
TIME_CHECKER_PID=$!

# Setup repository
cd /workspace
if [ ! -d "adaptive-llm-agents" ]; then
    git clone https://github.com/nullbio/adaptive-llm-agents.git
fi
cd adaptive-llm-agents

# Install dependencies
pip install -r requirements.txt || exit 1

# Get node rank
NODE_RANK=${{NODE_RANK:-0}}
echo "Starting Node $NODE_RANK of $WORLD_SIZE"

# Create monitoring log
LOG_FILE="/workspace/node_${{NODE_RANK}}_training.log"

# Run training with logging
python -u train_distributed.py 2>&1 | tee $LOG_FILE &
TRAINING_PID=$!

# Monitor training process
while kill -0 $TRAINING_PID 2>/dev/null; do
    # Check for common errors in log
    if grep -q "CUDA out of memory" $LOG_FILE; then
        echo "❌ Out of memory error detected!"
        kill $TRAINING_PID
        touch /workspace/training_oom
        exit 1
    fi
    
    if grep -q "RuntimeError\\|ValueError\\|KeyError" $LOG_FILE; then
        echo "❌ Python error detected!"
        # Don't exit immediately - might be recoverable
    fi
    
    # Show progress
    LAST_LOSS=$(grep -oP "Loss: \\K[0-9.]+" $LOG_FILE | tail -1)
    if [ ! -z "$LAST_LOSS" ]; then
        echo "📊 Current loss: $LAST_LOSS"
    fi
    
    sleep 30
done

# Check exit status
wait $TRAINING_PID
EXIT_CODE=$?

# Cleanup
kill $TIME_CHECKER_PID 2>/dev/null

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    touch /workspace/training_completed
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    touch /workspace/training_failed
fi

# Save logs to persistent storage
cp $LOG_FILE /workspace/final_logs/

exit $EXIT_CODE
"""
    
    def _get_master_ip(self, nodes: list):
        """Get internal IP of master node."""
        if not nodes:
            return "localhost"
            
        master = nodes[0]
        runtime = master.get("runtime")
        if runtime and runtime.get("ports"):
            for port in runtime.get("ports", []):
                if not port.get("isIpPublic", True):
                    return port["ip"]
                    
            # Fallback to first available IP
            for port in runtime.get("ports", []):
                if "ip" in port:
                    return port["ip"]
                
        # If no runtime info yet, use placeholder
        return "MASTER_IP_PENDING"
    
    def cleanup_cluster(self, node_ids: list):
        """Cleanup all nodes in cluster."""
        if self.cleanup_done:
            return
            
        self.cleanup_done = True
        print(f"\n🧹 Cleaning up {len(node_ids)} nodes...")
        
        # Calculate final cost
        elapsed_hours = (time.time() - self.start_time) / 3600
        total_cost = elapsed_hours * self.cost_per_hour
        
        print(f"⏱️  Total runtime: {elapsed_hours:.2f} hours")
        print(f"💰 Total cost: ${total_cost:.2f}")
        
        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        
        for node_id in node_ids:
            try:
                variables = {"input": {"podId": node_id}}
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json={"query": mutation, "variables": variables},
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"✅ Terminated node: {node_id[:8]}...")
                else:
                    print(f"❌ Failed to terminate: {node_id[:8]}...")
                    
            except Exception as e:
                print(f"❌ Error terminating {node_id[:8]}: {e}")
                
        print("\n✅ Cleanup complete!")
        
    def save_cluster_info(self, nodes: list):
        """Save cluster information for recovery."""
        cluster_info = {
            "created_at": datetime.now().isoformat(),
            "max_hours": self.max_hours,
            "max_cost": self.max_cost,
            "cost_per_hour": self.cost_per_hour,
            "nodes": nodes,
            "node_ids": self.node_ids
        }
        
        with open("cluster_info_safe.json", "w") as f:
            json.dump(cluster_info, f, indent=2)
            
        print(f"\n💾 Cluster info saved to cluster_info_safe.json")


def main():
    parser = argparse.ArgumentParser(description="Safe RunPod cluster deployment")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--num-nodes", type=int, default=5, help="Number of nodes")
    parser.add_argument("--max-hours", type=float, default=6.0, help="Max hours before auto-cleanup")
    parser.add_argument("--max-cost", type=float, default=20.0, help="Max cost in USD before auto-cleanup")
    parser.add_argument("--cleanup-only", action="store_true", help="Cleanup existing cluster")
    parser.add_argument("--gpu-type", default="NVIDIA RTX A4000", help="GPU type (e.g. 'NVIDIA RTX A4000', 'NVIDIA RTX 3090')")
    args = parser.parse_args()
    
    deployer = SafeRunPodDeployer(
        api_key=args.api_key,
        max_hours=args.max_hours,
        max_cost=args.max_cost,
        gpu_type=args.gpu_type
    )
    
    if args.cleanup_only:
        # Load and cleanup existing cluster
        if os.path.exists("cluster_info_safe.json"):
            with open("cluster_info_safe.json", "r") as f:
                info = json.load(f)
                deployer.cleanup_cluster(info["node_ids"])
        else:
            print("No cluster info found")
        return
        
    try:
        # Create cluster
        nodes = deployer.create_cluster_with_timeout(args.num_nodes)
        
        if not nodes:
            print("❌ Failed to create cluster")
            return
            
        # Save info for recovery
        deployer.save_cluster_info(nodes)
        
        # Wait for nodes to be ready
        print("\n⏳ Waiting for nodes to be ready...")
        print("Note: Nodes need time to start up. SSH info will be available once ready.")
        
        # Save deployment script for later use
        training_script = deployer._generate_monitored_script(nodes)
        with open("monitored_train.sh", "w") as f:
            f.write(training_script)
        print("\n📝 Training script saved to monitored_train.sh")
        
        # Give basic instructions
        print("\n📋 Once nodes are ready, connect to each node and run:")
        print("bash < monitored_train.sh")
        
        print("\n" + "="*50)
        print("🚀 CLUSTER READY WITH AUTO-CLEANUP")
        print("="*50)
        print(f"⏰ Will auto-terminate after {args.max_hours} hours")
        print(f"💰 Will auto-terminate at ${args.max_cost}")
        print(f"📊 Cost: ${deployer.cost_per_hour:.2f}/hour")
        print("\n⚠️  Keep this script running for auto-cleanup!")
        print("Press Ctrl+C to force cleanup now")
        
        # Keep running for monitoring
        while not deployer.cleanup_done:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted - cleaning up...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Always cleanup
        if deployer.node_ids:
            deployer.cleanup_cluster(deployer.node_ids)


if __name__ == "__main__":
    main()
