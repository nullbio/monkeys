#!/usr/bin/env python3
"""Debug pod information to see actual data structure."""

import json
import requests
import argparse

def debug_pods(api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Try to get all possible fields
    query = """
    query GetPods {
        myself {
            pods {
                id
                name
                desiredStatus
                imageName
                runtime {
                    uptimeInSeconds
                    gpus {
                        id
                        gpuUtilPercent
                    }
                    ports {
                        ip
                        isIpPublic  
                        privatePort
                        publicPort
                    }
                }
                machine {
                    podHostId
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
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()
    
    debug_pods(args.api_key)