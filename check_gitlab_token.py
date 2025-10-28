#!/usr/bin/env python3
"""
Check GitLab token permissions and access
"""

import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

async def check_gitlab_token():
    """Check GitLab token permissions"""
    token = os.getenv('GITLAB_TOKEN')
    gitlab_url = os.getenv('GITLAB_URL', 'https://gitlab.com')
    
    if not token:
        print("❌ GITLAB_TOKEN not found in environment variables")
        return
    
    print(f"🔍 Checking token: {token[:12]}...")
    print(f"🌐 GitLab URL: {gitlab_url}")
    
    headers = {
        "PRIVATE-TOKEN": token,
        "Content-Type": "application/json"
    }
    
    base_url = f"{gitlab_url}/api/v4"
    
    async with httpx.AsyncClient() as client:
        try:
            # Check user info
            print("\n📋 Checking user info...")
            response = await client.get(f"{base_url}/user", headers=headers, timeout=10.0)
            if response.status_code == 200:
                user = response.json()
                print(f"✅ User: {user.get('name', 'N/A')} (@{user.get('username', 'N/A')})")
                print(f"✅ Email: {user.get('email', 'N/A')}")
            else:
                print(f"❌ User info failed: {response.status_code} - {response.text}")
                return
            
            # Check projects access
            print("\n📁 Checking projects access...")
            response = await client.get(f"{base_url}/projects", headers=headers, timeout=10.0)
            if response.status_code == 200:
                projects = response.json()
                print(f"✅ Can access {len(projects)} projects")
                if projects:
                    print("📋 First 5 projects:")
                    for project in projects[:5]:
                        print(f"   • {project.get('name')} (ID: {project.get('id')})")
                        print(f"     URL: {project.get('web_url')}")
            else:
                print(f"❌ Projects access failed: {response.status_code}")
            
            # Check specific project if URL provided
            repo_url = "https://jitlocker.changepond.com/gitlab-instance-b7d10beb/asisnewcr.git"
            if repo_url:
                print(f"\n🎯 Checking specific project: {repo_url}")
                
                # Extract project path
                import re
                match = re.match(r'https?://[^/]+/(.+?)(?:\.git)?/?$', repo_url)
                if match:
                    project_path = match.group(1).rstrip('.git')
                    print(f"📂 Project path: {project_path}")
                    
                    from urllib.parse import quote
                    encoded_path = quote(project_path, safe='')
                    
                    response = await client.get(f"{base_url}/projects/{encoded_path}", headers=headers, timeout=10.0)
                    if response.status_code == 200:
                        project = response.json()
                        print(f"✅ Project found: {project.get('name')} (ID: {project.get('id')})")
                        print(f"✅ Visibility: {project.get('visibility')}")
                        
                        # Check repository access
                        project_id = project.get('id')
                        response = await client.get(f"{base_url}/projects/{project_id}/repository/tree", headers=headers, timeout=10.0)
                        if response.status_code == 200:
                            files = response.json()
                            print(f"✅ Repository access: Can see {len(files)} files/folders")
                        else:
                            print(f"❌ Repository access failed: {response.status_code} - {response.text}")
                    else:
                        print(f"❌ Project access failed: {response.status_code} - {response.text}")
                        if response.status_code == 401:
                            print("💡 Token may not have sufficient permissions")
                        elif response.status_code == 404:
                            print("💡 Project not found or no access")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(check_gitlab_token())