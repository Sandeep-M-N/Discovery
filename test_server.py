#!/usr/bin/env python3
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Test the server initialization
from server import gitlab

async def test_server():
    print("Testing GitLab server...")
    
    try:
        # Test list repository
        print("Testing list_repository...")
        files = await gitlab.list_repository()
        print(f"Success! Found {len(files)} items in root directory")
        for f in files[:5]:  # Show first 5
            print(f"  - {f['name']} ({f['type']})")
        
        # Test read file if any files exist
        if files:
            first_file = next((f for f in files if f['type'] == 'blob'), None)
            if first_file:
                print(f"\nTesting read_file with: {first_file['name']}")
                content = await gitlab.get_raw_file(first_file['path'])
                print(f"Success! File size: {len(content)} characters")
                print(f"First 100 chars: {content[:100]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_server())