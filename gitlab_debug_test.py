#!/usr/bin/env python3
"""
GitLab Configuration Test Script
Run this to debug your GitLab connection
"""

import os
import httpx
import asyncio
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()

async def test_gitlab_connection():
    """Test GitLab API connection and find correct project identifier"""
    
    # Get configuration
    gitlab_token = os.getenv("GITLAB_TOKEN")
    gitlab_url_input = "https://jitlocker.changepond.com/gitlab-instance-b7d10beb/asisnewcr.git"
    
    if not gitlab_token:
        print("❌ GITLAB_TOKEN not found in environment")
        return
    
    print("=" * 80)
    print("GitLab Configuration Test")
    print("=" * 80)
    print(f"Input URL: {gitlab_url_input}")
    print(f"Token: {gitlab_token[:10]}...{gitlab_token[-4:]}")
    print()
    
    # Parse URL
    import re
    match = re.match(r'https?://([^/]+)/(.+?)(?:\.git)?/?$', gitlab_url_input)
    if match:
        domain = match.group(1)
        project_path = match.group(2).rstrip('.git').rstrip('/')
        base_url = f"https://{domain}"
        print(f"✅ Parsed URL:")
        print(f"   Base URL: {base_url}")
        print(f"   Project Path: {project_path}")
        print()
    else:
        print("❌ Could not parse URL")
        return
    
    # Setup API client
    api_base = f"{base_url}/api/v4"
    headers = {
        "PRIVATE-TOKEN": gitlab_token,
        "Content-Type": "application/json"
    }
    
    print("=" * 80)
    print("Testing Different Project Identifiers")
    print("=" * 80)
    
    # Test 1: Try with URL-encoded full path
    print("\n📝 Test 1: Using URL-encoded path")
    encoded_path = quote(project_path, safe='')
    test_url = f"{api_base}/projects/{encoded_path}"
    print(f"   Testing: {test_url}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(test_url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ SUCCESS!")
                print(f"   Project ID: {data['id']}")
                print(f"   Project Name: {data['name']}")
                print(f"   Path with Namespace: {data['path_with_namespace']}")
                print(f"\n   Use this in your .env:")
                print(f"   GITLAB_PROJECT_PATH={data['path_with_namespace']}")
                return data['id'], data['path_with_namespace']
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 2: List all projects to find the correct one
    print("\n📝 Test 2: Listing all accessible projects")
    list_url = f"{api_base}/projects?membership=true&per_page=100"
    print(f"   Testing: {list_url}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(list_url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                projects = response.json()
                print(f"   ✅ Found {len(projects)} accessible projects:")
                print()
                
                for proj in projects:
                    print(f"   📁 {proj['name']}")
                    print(f"      ID: {proj['id']}")
                    print(f"      Path: {proj['path_with_namespace']}")
                    print(f"      URL: {proj['web_url']}")
                    
                    # Check if this matches our target
                    if 'asisnewcr' in proj['path'].lower() or 'asisnewcr' in proj['name'].lower():
                        print(f"      ⭐ POTENTIAL MATCH!")
                        print(f"\n      Use this configuration:")
                        print(f"      GITLAB_PROJECT_ID={proj['id']}")
                        print(f"      GITLAB_PROJECT_PATH={proj['path_with_namespace']}")
                    print()
                
                return
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 3: Try different path variations
    print("\n📝 Test 3: Trying path variations")
    path_variations = [
        project_path,
        project_path.split('/')[-1],  # Just the last part
        f"gitlab-instance-b7d10beb/{project_path}",
    ]
    
    for variation in path_variations:
        encoded = quote(variation, safe='')
        test_url = f"{api_base}/projects/{encoded}"
        print(f"\n   Trying: {variation}")
        print(f"   URL: {test_url}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(test_url, headers=headers, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ SUCCESS with this path!")
                    print(f"   Project ID: {data['id']}")
                    print(f"   Correct Path: {data['path_with_namespace']}")
                    return data['id'], data['path_with_namespace']
                else:
                    print(f"   ❌ {response.status_code}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    print("1. Check if your token has 'read_api' and 'read_repository' scopes")
    print("2. Verify you have access to this project in GitLab web interface")
    print("3. The project might be in a group/namespace - check the full path")
    print("4. Try accessing the GitLab web UI and copy the exact project path from URL")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_gitlab_connection())