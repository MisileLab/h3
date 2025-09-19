import os
import vastai
import subprocess
import re
from dotenv import load_dotenv
import argparse

load_dotenv()

# --- Configuration ---
# IMPORTANT: Set your Vast.ai API key in a .env file
# VAST_API_KEY='your_api_key_here'
API_KEY = os.environ.get("VAST_API_KEY")
print(f"DEBUG: API_KEY after load_dotenv: {API_KEY}")

# Server search criteria
MIN_VRAM = 80  # in GB

# Deployment configuration
GIT_REPO_URL = "https://gith.misile.xyz.git:/projects/exts/myhandnomusa/finetuning.git"
DOCKER_IMAGE = "pytorch/pytorch:latest"
ON_START_SCRIPT_NAME = "on-start.sh"
DISK_SPACE_GB = 30 # How much disk space to allocate in GB

def generate_on_start_script():
    """Generates the shell script that will be run on instance startup."""
    script_content = f"""#!/bin/bash

echo "--- Starting automatic deployment script ---"

# Clone the repository
echo "Cloning repository: {GIT_REPO_URL}"
# The provided git URL is non-standard. We'll try to clone it as is.
# The /workspace directory is the default working directory in many Vast.ai images.
git clone {GIT_REPO_URL} /workspace
cd /workspace
echo "Repository cloned."

# Install dependencies
echo "Installing dependencies with uv..."
pip install uv
uv sync
echo "Dependencies installed."

# Start the fine-tuning process in the background
echo "Starting fine-tuning process..."
nohup uv run python finetuning.py > finetune.log 2>&1 &

echo "--- Finetuning started in the background. ---"
echo "You can monitor the progress by checking the 'finetune.log' file."
echo "Use 'vastai logs <instance_id>' to see live logs."

"""
    # Make sure the script file is created in the current directory
    script_path = os.path.abspath(ON_START_SCRIPT_NAME)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    subprocess.run(["chmod", "+x", script_path])
    
    print(f"Generated startup script: {script_path}")
    return script_path

def find_cheapest_instance(client):
    """Finds the cheapest available and reliable instance that meets the criteria using the SDK."""
    print(f"Searching for instances with at least {MIN_VRAM}GB VRAM using SDK...")
    
    
    try:
        # The SDK expects a single query string with direct comparisons.
        # gpu_total_ram is in GB.
        query = f"gpu_total_ram >= {MIN_VRAM} rentable=true verified=true"
        print(f"DEBUG: Query string: {query}")
        
        
        instances = client.search_offers(
            query=query,
            order="dph_total"
        )
        print(f"DEBUG: Raw instances found: {len(instances) if instances else 0}")
        
        print(f"DEBUG: Raw instances content (first 5): {instances[:5]}") # Print first 5 for inspection
        

        if not instances:
            print("No instances found matching the criteria.")
            return None
        
        return instances[0]
        
    except Exception as e:
        print(f"An error occurred while searching for instances: {e}")
        return None

def main():
    """Main function to find, confirm, and deploy the instance."""
    parser = argparse.ArgumentParser(description="Deploy a fine-tuning job to Vast.ai.")
    parser.add_argument('-y', '--yes', action='store_true', help='Automatically confirm instance rental.')
    args = parser.parse_args()

    if not API_KEY:
        print("Error: VAST_API_KEY environment variable is not set.")
        print("Please set it by running: `echo VAST_API_KEY=your_key > .env`")
        return

    # Initialize the VastAI client
    try:
        client = vastai.VastAI(api_key=API_KEY)
        print("DEBUG: VastAI client initialized successfully.")
        
    except Exception as e:
        print(f"Error initializing VastAI client: {e}")
        print("Please ensure your VAST_API_KEY is valid and you have network connectivity.")
        return

    print("DEBUG: Calling find_cheapest_instance...")
    
    try:
        cheapest = find_cheapest_instance(client)
    except Exception as e:
        print(f"Error during instance search: {e}")
        return

    if not cheapest:
        return

    print("\n--- Cheapest Instance Found (Including Unverified) ---")
    print(f"  ID: {cheapest['id']}")
    print(f"  GPU: {cheapest['num_gpus']}x {cheapest['gpu_name']}")
    print(f"  VRAM: {cheapest['gpu_total_ram'] / 1024:.1f} GB")
    print(f"  Price: ${cheapest['dph_total']:.4f}/hour")
    print(f"  Reliability: {cheapest['reliability2'] * 100:.2f}%")
    print(f"  Verified: {cheapest.get('verified', False)}")
    print(f"  Location: {cheapest['geolocation']}")
    print("------------------------------------------------------")
    
    if not cheapest.get('verified', False):
        print("** Warning: This is an unverified instance. It may have lower reliability. **")
        

    if args.yes:
        print("Auto-confirming instance rental due to -y option.")
        confirm = 'y'
    else:
        confirm = input("Do you want to rent this instance? (y/n): ")

    if confirm.lower() != 'y':
        print("Deployment cancelled.")
        return

    print("\n--- Starting Deployment ---")
    
    # 1. Generate the startup script
    script_path = generate_on_start_script()

    # 2. Rent the instance using vastai CLI
    print(f"Renting instance {cheapest['id']} using CLI...")
    try:        
        command = [
            "vastai", "create", "instance", str(cheapest['id']),
            "--image", DOCKER_IMAGE,
            "--onstart", script_path,  # Pass the file path, not the content
            "--disk", str(DISK_SPACE_GB),
            "--api-key", API_KEY # Pass API key directly to CLI
        ]
        
        print(f"DEBUG: Executing command: {' '.join(command)}")
        
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("CLI Output:")
        print(result.stdout)
        
        if result.stderr:
            print("CLI Error:")
            print(result.stderr)

        # Check for successful creation (CLI output might vary)
        if "new_contract" in result.stdout or "Instance created" in result.stdout:
            # Attempt to parse the instance ID from the output
            new_id_match = re.search(r'new_contract\': (\d+)', result.stdout)
            new_id = new_id_match.group(1) if new_id_match else "Unknown"

            print("\n--- Deployment Successful! ---")
            print(f"Instance {new_id} is being set up.")
            print("The startup script will clone the repo and start finetuning.")
            print(f"To monitor logs, use: vastai logs {new_id}")
            print(f"To connect, use: vastai ssh {new_id}")
            print(f"To destroy the instance later, use: vastai destroy instance {new_id}")
            
        else:
            print("Failed to create instance. CLI output did not confirm creation.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during instance creation (CLI): {e}")
        print(f"Stderr: {e.stderr}")
        
    except Exception as e:
        print(f"An unexpected error occurred during instance creation: {e}")
    
    finally:
        # Clean up the temporary script file
        if os.path.exists(script_path):
            try:
                os.remove(script_path)
                print(f"Cleaned up temporary script file: {script_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary script file {script_path}: {e}")

if __name__ == "__main__":
    main()
