# RunPod Cluster Training Guide

## Quick Start (One-Click Deployment)

### 1. Get RunPod API Key

- Sign up at [runpod.io](https://runpod.io)
- Go to Settings → API Keys
- Create new API key

### 2. Deploy Cluster

```bash
# Install requirements
pip install requests paramiko pyyaml

# Deploy 5-node cluster
python deploy_runpod_cluster.py --api-key YOUR_API_KEY --num-nodes 5
```

This will:

- Create 5 RTX 4090 nodes (~$2.20/hour total)
- Set up distributed training automatically
- Start training all 5 agents in parallel
- Complete in ~3-4 hours

### 3. Monitor Progress

The script provides:

- SSH commands for each node
- Jupyter Lab URLs
- Monitoring instructions

### 4. Cleanup When Done

```bash
python deploy_runpod_cluster.py --api-key YOUR_API_KEY --cleanup
```

## Manual Setup (If Automated Fails)

### Step 1: Create Cluster on RunPod UI

1. Go to RunPod Console → Clusters
2. Click "Create Cluster"
3. Select:
   - 5 nodes
   - RTX 4090
   - PyTorch 2.0 template
   - 50GB storage per node

### Step 2: Connect to Master Node (Node 0)

```bash
ssh root@[master-ip] -p [port]
```

### Step 3: Setup Repository

```bash
cd /workspace
git clone https://github.com/nullbio/adaptive-llm-agents.git
cd adaptive-llm-agents
pip install -r requirements.txt
```

### Step 4: Get Master IP

```bash
hostname -I | awk '{print $1}'
```

### Step 5: Launch on ALL Nodes

On each node, run:

```bash
export NODE_RANK=0  # Change to 1,2,3,4 for other nodes
export MASTER_ADDR=[master-ip-from-step-4]
export MASTER_PORT=29500
export WORLD_SIZE=5

cd /workspace/adaptive-llm-agents

torchrun \
  --nproc_per_node=1 \
  --nnodes=5 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_distributed.py
```

## Expected Timeline

- **Cluster Creation**: 5-10 minutes
- **Setup**: 10 minutes
- **Training**: 3-4 hours
- **Total**: ~4-5 hours

## Cost Breakdown

- 5x RTX 4090: $0.44 × 5 = $2.20/hour
- 5 hours total: ~$11
- **Much cheaper than single A100!**

## Troubleshooting

### Nodes Can't Communicate

- Ensure all nodes are in same region
- Check firewall allows port 29500
- Use internal IPs, not public

### Out of Memory

- Reduce batch size in config
- Enable gradient checkpointing

### Training Stalls

- Check all nodes started successfully
- Monitor with `nvidia-smi` on each node
- Check logs in `/workspace/adaptive-llm-agents/`

## Results

After training completes:

1. Models saved to `/workspace/checkpoints/`
2. Download with:

   ```bash
   scp -P [port] root@[node-ip]:/workspace/checkpoints/* ./local_checkpoints/
   ```

3. Or use RunPod's web interface to download

## Advanced Options

### Use H100 Nodes (Faster but Pricier)

```bash
# Edit deploy script to use H100
python deploy_runpod_cluster.py --api-key KEY --gpu-type "NVIDIA H100" --num-nodes 5
```

Cost: ~$20/hour, completes in ~1 hour

### Train More Agents

```bash
# 10 agents on 10 nodes
python deploy_runpod_cluster.py --api-key KEY --num-nodes 10
```

### Custom Configuration

Edit `config.yaml` before deployment:

- Increase `num_agents`
- Adjust `bias_strength`
- Change training parameters
