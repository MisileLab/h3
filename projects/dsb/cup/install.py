#!/usr/bin/env python3
"""
Installation script for PDF to CSV Converter.

This script helps users set up the environment and dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_uv():
    """Check if uv is installed."""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ uv is not installed")
    print("   Install uv: https://docs.astral.sh/uv/getting-started/installation/")
    return False

def install_dependencies():
    """Install project dependencies using uv."""
    print("\n📦 Installing dependencies...")
    try:
        result = subprocess.run(['uv', 'sync'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_openai_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        if api_key.startswith('sk-') and len(api_key) > 20:
            print("✅ OpenAI API key is set")
            return True
        else:
            print("⚠️  OpenAI API key format appears invalid")
            print("   Key should start with 'sk-' and be longer than 20 characters")
            return False
    else:
        print("⚠️  OpenAI API key not set")
        print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False

def create_example_env():
    """Create example environment file."""
    env_content = """# OpenAI API Key
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-api-key-here

# Optional: Performance settings for GPU
# TORCH_DEVICE=cuda
# RECOGNITION_BATCH_SIZE=512
# DETECTOR_BATCH_SIZE=36
# TABLE_REC_BATCH_SIZE=64

# Optional: Performance settings for CPU
# RECOGNITION_BATCH_SIZE=32
# DETECTOR_BATCH_SIZE=6
# TABLE_REC_BATCH_SIZE=8
"""
    
    env_file = Path('.env.example')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env.example file")
        print("   Copy it to .env and add your OpenAI API key")

def show_next_steps():
    """Show next steps for the user."""
    print("\n🎉 Installation completed!")
    print("\n📋 Next steps:")
    print("\nOption 1: Using Infisical (Recommended)")
    print("1. Set up Infisical for secure secrets management:")
    print("   python setup_infisical.py")
    print("2. Convert your first PDF:")
    print("   infisical run -- python pdf_to_csv.py convert your_document.pdf")
    
    print("\nOption 2: Using Environment Variables")
    print("1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("   # Or copy .env.example to .env and edit it")
    print("2. Convert your first PDF:")
    print("   python pdf_to_csv.py convert your_document.pdf")
    
    print("\n3. Test the installation:")
    print("   python pdf_to_csv.py info")
    print("\n4. Get help:")
    print("   python pdf_to_csv.py --help")
    print("\n📚 For more information, see README.md")

def main():
    """Main installation function."""
    print("🚀 PDF to CSV Converter - Installation")
    print("=" * 50)
    
    # Check requirements
    checks_passed = True
    
    if not check_python_version():
        checks_passed = False
    
    if not check_uv():
        checks_passed = False
    
    if not checks_passed:
        print("\n❌ Please fix the issues above and run this script again")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check OpenAI key
    check_openai_key()
    
    # Create example environment file
    create_example_env()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main() 