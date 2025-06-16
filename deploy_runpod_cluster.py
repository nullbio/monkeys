#!/usr/bin/env python3
"""
One-click deployment script for RunPod multi-node cluster training.
Automatically sets up and coordinates training across all nodes.
"""

import os
import json
import time
import subprocess
import requests
from pathlib import Path
import yaml
import argparse


class RunPodClusterDeployer:
    """Deploy and manage distributed training on RunPod cluster."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
    def create_cluster(self, num_nodes: int = 5):
        """Create a RunPod cluster with specified nodes."""
        print(f"Creating cluster with {num_nodes} nodes...")
        
        # GraphQL mutation to create cluster
        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                machineId
                costPerHr
                gpuCount
                vcpuCount
                memoryInGb
            }
        }
        """
        
        # Configuration for each node
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuCount": 1,
                "volumeInGb": 50,
                "containerDiskInGb": 50,
                "minVcpuCount": 8,
                "minMemoryInGb": 24,
                "gpuTypeId": "NVIDIA RTX 4090",
                "name": f"agent-trainer-node",
                "dockerArgs": "",
                "ports": "8888,22",
                "volumeMountPath": "/workspace",
                "env": [
                    {"key": "JUPYTER_PASSWORD", "value": "adaptive_agents"},
                    {"key": "PUBLIC_KEY", "value": ""},  # Add your SSH key
                ],
                "templateId": "runpod-pytorch-2.0.0",
                "networkVolumeId": None
            }
        }
        
        nodes = []
        for i in range(num_nodes):
            variables["input"]["name"] = f"agent-trainer-node-{i}"
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={"query": mutation, "variables": variables}
            )
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result:
                    node_info = result["data"]["podFindAndDeployOnDemand"]
                    nodes.append(node_info)
                    print(f"Created node {i}: {node_info['name']} - ${node_info['costPerHr']}/hr")
                else:
                    print(f"Error creating node {i}: {result}")
            else:
                print(f"HTTP Error: {response.status_code}")
            
            time.sleep(2)  # Rate limiting
        
        return nodes
    
    def wait_for_nodes(self, node_ids: list, timeout: int = 300):
        """Wait for all nodes to be ready."""
        print("Waiting for nodes to be ready...")
        
        query = """
        query GetPods($input: PodFilter) {
            myself {
                pods(input: $input) {
                    id
                    name
                    runtime {
                        status
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
        
        start_time = time.time()
        ready_nodes = []
        
        while len(ready_nodes) < len(node_ids) and (time.time() - start_time) < timeout:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={"query": query, "variables": {"input": {}}}
            )
            
            if response.status_code == 200:
                result = response.json()
                pods = result["data"]["myself"]["pods"]
                
                for pod in pods:
                    if pod["id"] in node_ids and pod["runtime"]["status"] == "RUNNING":
                        if pod["id"] not in [n["id"] for n in ready_nodes]:
                            ready_nodes.append(pod)
                            print(f"Node {pod['name']} is ready!")
            
            if len(ready_nodes) < len(node_ids):
                time.sleep(10)
        
        return ready_nodes
    
    def setup_master_node(self, nodes: list):
        """Configure the master node for distributed training."""
        master_node = nodes[0]
        master_ip = None
        
        # Get internal IP of master
        for port in master_node["runtime"]["ports"]:
            if not port["isIpPublic"]:
                master_ip = port["ip"]
                break
        
        if not master_ip:
            # Fallback to public IP
            for port in master_node["runtime"]["ports"]:
                if port["isIpPublic"]:
                    master_ip = port["ip"]
                    break
        
        return master_ip
    
    def generate_launch_script(self, nodes: list, master_ip: str):
        """Generate the launch script for all nodes."""
        
        script_content = f"""#!/bin/bash
# Auto-generated RunPod cluster training script

# Configuration
export MASTER_ADDR={master_ip}
export MASTER_PORT=29500
export WORLD_SIZE={len(nodes)}
export CUDA_VISIBLE_DEVICES=0

# Clone repository (if not exists)
if [ ! -d "/workspace/adaptive-llm-agents" ]; then
    cd /workspace
    git clone https://github.com/nullbio/adaptive-llm-agents.git
    cd adaptive-llm-agents
    pip install -r requirements.txt
else
    cd /workspace/adaptive-llm-agents
    git pull
fi

# Get node rank from hostname or environment
if [ -z "$NODE_RANK" ]; then
    # Extract node number from hostname
    NODE_RANK=$(hostname | grep -oE '[0-9]+$' || echo "0")
fi

export RANK=$NODE_RANK

echo "Starting training on Node $NODE_RANK of $WORLD_SIZE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Launch distributed training
python -m torch.distributed.launch \\
    --nproc_per_node=1 \\
    --nnodes=$WORLD_SIZE \\
    --node_rank=$NODE_RANK \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    train_distributed.py

# Alternative: Use torchrun (newer)
# torchrun \\
#     --nproc_per_node=1 \\
#     --nnodes=$WORLD_SIZE \\
#     --node_rank=$NODE_RANK \\
#     --master_addr=$MASTER_ADDR \\
#     --master_port=$MASTER_PORT \\
#     train_distributed.py
"""
        
        return script_content
    
    def deploy_to_nodes(self, nodes: list, script_content: str):
        """Deploy the training script to all nodes."""
        print("\nDeploying to all nodes...")
        
        # Save script locally first
        with open("cluster_train.sh", "w") as f:
            f.write(script_content)
        
        # Instructions for manual deployment
        print("\n" + "="*50)
        print("DEPLOYMENT INSTRUCTIONS:")
        print("="*50)
        
        for i, node in enumerate(nodes):
            ssh_command = None
            for port in node["runtime"]["ports"]:
                if port["type"] == "ssh" and port["isIpPublic"]:
                    ssh_command = f"ssh root@{port['ip']} -p {port['publicPort']}"
                    break
            
            print(f"\nNode {i} ({node['name']}):")
            print(f"1. Connect: {ssh_command}")
            print(f"2. Run: export NODE_RANK={i}")
            print(f"3. Paste and run the cluster_train.sh script")
        
        print("\n" + "="*50)
        print("AUTOMATED DEPLOYMENT:")
        print("="*50)
        print("\nRun this command to deploy to all nodes automatically:")
        print(f"python deploy_to_all_nodes.py --cluster-id {nodes[0]['id'][:8]}")
        
        # Generate automated deployment script
        self._generate_auto_deploy_script(nodes)
    
    def _generate_auto_deploy_script(self, nodes: list):
        """Generate script for automated deployment."""
        
        auto_deploy = """#!/usr/bin/env python3
import paramiko
import threading
import time

def deploy_to_node(node_info, node_rank, script_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(
            hostname=node_info['ip'],
            port=node_info['port'],
            username='root',
            password='runpod'  # Default RunPod password
        )
        
        # Upload script
        sftp = ssh.open_sftp()
        sftp.put(script_path, '/workspace/train.sh')
        sftp.chmod('/workspace/train.sh', 0o755)
        sftp.close()
        
        # Execute
        stdin, stdout, stderr = ssh.exec_command(
            f'export NODE_RANK={node_rank} && cd /workspace && ./train.sh'
        )
        
        # Print output
        for line in stdout:
            print(f"Node {node_rank}: {line.strip()}")
            
    except Exception as e:
        print(f"Error on node {node_rank}: {e}")
    finally:
        ssh.close()

# Node connection info
nodes = %s

# Deploy in parallel
threads = []
for i, node in enumerate(nodes):
    t = threading.Thread(target=deploy_to_node, args=(node, i, 'cluster_train.sh'))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("Deployment complete!")
""" % str([{"ip": n["runtime"]["ports"][0]["ip"], 
            "port": n["runtime"]["ports"][0]["publicPort"]} 
           for n in nodes if n["runtime"]["ports"]])
        
        with open("deploy_to_all_nodes.py", "w") as f:
            f.write(auto_deploy)
        
        print("\nGenerated: deploy_to_all_nodes.py")
    
    def monitor_training(self, nodes: list):
        """Monitor training progress across all nodes."""
        print("\n" + "="*50)
        print("MONITORING:")
        print("="*50)
        
        print("\nTo monitor training progress:")
        print("1. SSH into any node")
        print("2. Run: tail -f /workspace/adaptive-llm-agents/training.log")
        print("\nTo check GPU usage:")
        print("Run: watch -n 1 nvidia-smi")
        
        # Generate monitoring dashboard URL if available
        for node in nodes:
            for port in node["runtime"]["ports"]:
                if port["privatePort"] == 8888:  # Jupyter port
                    print(f"\nJupyter Lab: http://{port['ip']}:{port['publicPort']}")
                    print("Password: adaptive_agents")
                    break
    
    def cleanup_cluster(self, node_ids: list):
        """Terminate all nodes in the cluster."""
        print("\nCleaning up cluster...")
        
        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        
        for node_id in node_ids:
            variables = {"input": {"podId": node_id}}
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={"query": mutation, "variables": variables}
            )
            
            if response.status_code == 200:
                print(f"Terminated node: {node_id}")
            else:
                print(f"Error terminating node {node_id}")


def main():
    parser = argparse.ArgumentParser(description="Deploy RunPod cluster for distributed training")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--num-nodes", type=int, default=5, help="Number of nodes")
    parser.add_argument("--deploy-only", action="store_true", help="Skip cluster creation")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup existing cluster")
    args = parser.parse_args()
    
    deployer = RunPodClusterDeployer(args.api_key)
    
    if args.cleanup:
        # Load node IDs from file
        if os.path.exists("cluster_info.json"):
            with open("cluster_info.json", "r") as f:
                cluster_info = json.load(f)
                node_ids = [n["id"] for n in cluster_info["nodes"]]
                deployer.cleanup_cluster(node_ids)
        return
    
    if not args.deploy_only:
        # Create cluster
        nodes = deployer.create_cluster(args.num_nodes)
        
        # Save cluster info
        with open("cluster_info.json", "w") as f:
            json.dump({"nodes": nodes}, f, indent=2)
        
        # Wait for nodes
        node_ids = [n["id"] for n in nodes]
        ready_nodes = deployer.wait_for_nodes(node_ids)
    else:
        # Load existing cluster info
        with open("cluster_info.json", "r") as f:
            cluster_info = json.load(f)
            ready_nodes = cluster_info["nodes"]
    
    # Setup master and generate scripts
    master_ip = deployer.setup_master_node(ready_nodes)
    script_content = deployer.generate_launch_script(ready_nodes, master_ip)
    
    # Deploy
    deployer.deploy_to_nodes(ready_nodes, script_content)
    
    # Monitor
    deployer.monitor_training(ready_nodes)
    
    print("\n" + "="*50)
    print("CLUSTER READY!")
    print("="*50)
    print(f"\nTotal cost: ${sum(n['costPerHr'] for n in ready_nodes):.2f}/hour")
    print("\nTo terminate cluster when done:")
    print(f"python deploy_runpod_cluster.py --api-key {args.api_key} --cleanup")


if __name__ == "__main__":
    main()
