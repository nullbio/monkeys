#!/usr/bin/env python3
"""Test minimal RunPod pod creation."""

import requests
import json
import argparse

def test_create_pod(api_key):
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {api_key}"
    }
    
    # Try minimal configuration
    mutation = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            costPerHr
        }
    }
    """
    
    # Minimal required fields
    variables = {
        "input": {
            "cloudType": "SECURE",
            "gpuCount": 1,
            "gpuTypeId": "NVIDIA RTX A6000",
            "name": "test-pod",
            "imageName": "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "volumeInGb": 0,  # No persistent volume
            "containerDiskInGb": 10  # Minimal container disk
        }
    }
    
    print("Sending request with minimal config:")
    print(json.dumps(variables, indent=2))
    
    response = requests.post(
        "https://api.runpod.io/graphql",
        headers=headers,
        json={"query": mutation, "variables": variables}
    )
    
    print(f"\nStatus: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()
    
    test_create_pod(args.api_key)