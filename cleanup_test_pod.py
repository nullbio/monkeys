#!/usr/bin/env python3
"""Clean up test pod."""

import requests
import argparse

def cleanup_pod(api_key, pod_id):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    mutation = """
    mutation TerminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    
    variables = {"input": {"podId": pod_id}}
    
    response = requests.post(
        "https://api.runpod.io/graphql",
        headers=headers,
        json={"query": mutation, "variables": variables}
    )
    
    if response.status_code == 200:
        print(f"✅ Terminated pod: {pod_id}")
    else:
        print(f"❌ Failed to terminate pod: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--pod-id", default="b3gblwjl1et9po")
    args = parser.parse_args()
    
    cleanup_pod(args.api_key, args.pod_id)