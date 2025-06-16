#!/usr/bin/env python3
"""Check available GPU types on RunPod."""

import requests
import json
import argparse

def check_available_gpus(api_key):
    """Query RunPod for available GPU types."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Query to get GPU types
    query = """
    query GpuTypes {
        gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            lowestPrice {
                minimumBidPrice
                minimumSpotPrice
            }
        }
    }
    """
    
    response = requests.post(
        "https://api.runpod.io/graphql",
        headers=headers,
        json={"query": query}
    )
    
    if response.status_code == 200:
        result = response.json()
        if "data" in result and "gpuTypes" in result["data"]:
            gpu_types = result["data"]["gpuTypes"]
            
            print("\nAvailable GPU Types on RunPod:")
            print("=" * 80)
            
            # Sort by price
            gpu_types.sort(key=lambda x: x.get("lowestPrice", {}).get("minimumBidPrice", 999) or 999)
            
            for gpu in gpu_types:
                if gpu["secureCloud"]:
                    price = gpu.get("lowestPrice", {})
                    min_price = price.get("minimumBidPrice", "N/A") if price else "N/A"
                    
                    print(f"\nGPU: {gpu['displayName']}")
                    print(f"  ID: {gpu['id']}")
                    print(f"  Memory: {gpu['memoryInGb']} GB")
                    print(f"  Min Price: ${min_price}/hr" if min_price != "N/A" else "  Price: N/A")
                    print(f"  Secure Cloud: {'Yes' if gpu['secureCloud'] else 'No'}")
                    
            print("\n" + "=" * 80)
            print("\nUse the GPU ID (not display name) in your deployment script.")
            print("Example: --gpu-type 'NVIDIA GeForce RTX 3090'")
            
        else:
            print("Error:", result)
    else:
        print(f"HTTP Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check available RunPod GPU types")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    args = parser.parse_args()
    
    check_available_gpus(args.api_key)