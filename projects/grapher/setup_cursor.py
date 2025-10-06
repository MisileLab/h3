#!/usr/bin/env python3
"""
Setup script for configuring Code Knowledge Graph MCP server with Cursor IDE.

This script helps configure the MCP server for use with Cursor IDE.
"""

import json
import os
import sys
from pathlib import Path


def get_cursor_config_path():
    """Get the Cursor configuration directory path."""
    home = Path.home()
    
    # Try common locations for Cursor config
    cursor_paths = [
        home / ".cursor" / "mcp.json",
        home / ".config" / "cursor" / "mcp.json",
        home / "AppData" / "Roaming" / "Cursor" / "mcp.json" if sys.platform == "win32" else None,
        home / "Library" / "Application Support" / "Cursor" / "mcp.json" if sys.platform == "darwin" else None,
    ]
    
    for path in cursor_paths:
        if path and path.exists():
            return path
    
    # Default to ~/.cursor/mcp_servers.json
    return home / ".cursor" / "mcp_servers.json"


def setup_cursor_config():
    """Set up Cursor MCP configuration."""
    project_path = Path(__file__).parent.absolute()
    
    # Get the absolute path to the MCP server
    server_script = project_path / "mcp_server.py"
    
    # Create the MCP server configuration
    mcp_config = {
        "mcpServers": {
            "code-knowledge-graph": {
                "command": "uv",
                "args": [
                    "run",
                    "python",
                    str(server_script)
                ],
                "cwd": str(project_path),
                "env": {}
            }
        }
    }
    
    # Get Cursor config path
    cursor_config_path = get_cursor_config_path()
    
    # Create directory if it doesn't exist
    cursor_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing config if it exists
    existing_config = {}
    if cursor_config_path.exists():
        try:
            with open(cursor_config_path, 'r') as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read existing config at {cursor_config_path}")
    
    # Merge configurations
    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}
    
    existing_config["mcpServers"].update(mcp_config["mcpServers"])
    
    # Write the configuration
    try:
        with open(cursor_config_path, 'w') as f:
            json.dump(existing_config, f, indent=2)
        
        print(f"✅ Successfully configured MCP server for Cursor!")
        print(f"📁 Configuration written to: {cursor_config_path}")
        print(f"🔧 Server script: {server_script}")
        
        return True
    except IOError as e:
        print(f"❌ Error writing configuration: {e}")
        return False


def print_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🚀 CODE KNOWLEDGE GRAPH MCP SERVER SETUP")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Restart Cursor IDE")
    print("2. Open a Python project")
    print("3. Use the MCP tools in Cursor:")
    
    print("\n🛠️ Available Tools:")
    tools = [
        "analyze_project - Analyze a Python project",
        "ask_about_code - Ask questions about code",
        "get_project_structure - Get project overview",
        "find_functions - Find functions by criteria",
        "get_function_details - Get function information",
        "trace_function_calls - Trace call chains",
        "get_class_hierarchy - Get class inheritance"
    ]
    
    for tool in tools:
        print(f"  • {tool}")
    
    print("\n💡 Example Usage in Cursor:")
    print("1. First, analyze your project:")
    print("   Use the 'analyze_project' tool with your project path")
    
    print("\n2. Then ask questions:")
    print("   • 'What functions call the authenticate_user function?'")
    print("   • 'Show me all methods of the User class'")
    print("   • 'What functions are defined in the auth module?'")
    print("   • 'Trace the call chain for process_payment function'")
    
    print("\n🔍 Troubleshooting:")
    print("• Make sure uv is installed and in your PATH")
    print("• Check that the project path is correct")
    print("• Restart Cursor if tools don't appear")
    
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("🔧 Setting up Code Knowledge Graph MCP server for Cursor...")
    
    # Check if we're in the right directory
    if not Path("mcp_server.py").exists():
        print("❌ Error: mcp_server.py not found in current directory")
        print("Please run this script from the code-knowledge-graph project directory")
        sys.exit(1)
    
    # Set up the configuration
    if setup_cursor_config():
        print_instructions()
    else:
        print("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()