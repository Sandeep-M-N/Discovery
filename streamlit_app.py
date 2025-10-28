import asyncio
import json
import os
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

from langfuse import Langfuse, observe

langfuse = Langfuse(
    secret_key="sk-lf-fec877cf-e8b4-423a-aa60-8ca53357ffc1",
    public_key="pk-lf-fdb0bc03-7ab5-4def-99bd-d315391f8ab4",
    host="https://cloud.langfuse.com"
)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None


@observe()
async def process_gitlab_query(gitlab_url: str, query: str):
    try:
        # Validate GitLab token
        gitlab_token = os.getenv('GITLAB_TOKEN')
        
        if not gitlab_token:
            return {"error": "GITLAB_TOKEN environment variable is required"}
        
        # Use the known working configuration
        server_env = {
            "GITLAB_TOKEN": gitlab_token,
            "GITLAB_URL": "https://jitlocker.changepond.com",
            "GITLAB_PROJECT_ID": "335",
            "GITLAB_PROJECT_PATH": "gitlab-instance-b7d10beb/asisnewcr"
        }
        
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "server.py"],
            env=server_env
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools_result = await client.list_tools()
                
                openai_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("OPENAI_API_VERSION")
                )
                # Enhanced system prompt for deep analysis
                system_prompt = f"""You are a thorough GitLab repository analyst with access to these tools: {', '.join([t.name for t in tools_result.tools])}.

CRITICAL INSTRUCTIONS FOR COMPREHENSIVE ANALYSIS:
1. When exploring directories, you MUST recursively explore ALL subdirectories
2. For each directory found, call the tool again to list its contents
3. For each file found, read its contents to understand the codebase
4. Continue exploring until you've examined the entire repository structure
5. Don't stop after just the root directory - go deep into every folder
6. Track which directories you've already explored to avoid redundancy
7. Make multiple tool calls in sequence to build a complete picture

When asked to summarize a repository:
- First, list root directory contents
- Then, for each subdirectory found, recursively list its contents
- Read key configuration files (package.json, requirements.txt, etc.)
- Read source code files to understand functionality
- Analyze the overall structure and purpose

Be thorough and methodical. Don't summarize until you've fully explored the repository."""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze the GitLab repository and answer: {query}"}
                ]
                
                tool_calls = []
                
                for iteration in range(5):
                    try:
                        tools_schema = [{
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                            }
                        } for tool in tools_result.tools]
                        
                        response = openai_client.chat.completions.create(
                            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                            messages=messages,
                            tools=tools_schema,
                            max_tokens=4096,
                            temperature=0.7
                        )
                        
                        message = response.choices[0].message
                        
                        if not message.tool_calls:
                            messages.append({"role": "assistant", "content": message.content or ""})
                            break
                        
                        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        } for tc in message.tool_calls]})
                        
                        for tool_call in message.tool_calls:
                            try:
                                tool_name = tool_call.function.name
                                arguments = json.loads(tool_call.function.arguments)
                                
                                result = await client.call_tool(tool_name, arguments)
                                
                                tool_calls.append({
                                    "tool_name": tool_name,
                                    "arguments": arguments,
                                    "result": result.content
                                })
                                
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(result.content)
                                })
                            except Exception as tool_error:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": f"Error: {str(tool_error)}"
                                })
                    
                    except Exception as iter_error:
                        return {"error": f"Iteration {iteration} error: {str(iter_error)}"}
                
                if tool_calls:
                    messages.append({"role": "user", "content": "Based on the tool results above, provide a comprehensive and detailed analysis answering the original query."})
                    
                    final_response = openai_client.chat.completions.create(
                        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                        messages=messages,
                        temperature=0.2
                    )
                    final_answer = final_response.choices[0].message.content
                else:
                    final_answer = next((msg["content"] for msg in reversed(messages) if msg["role"] == "assistant" and msg.get("content")), "No response")
                
                return {
                    "tool_calls": tool_calls,
                    "final_answer": final_answer
                }
                
    except Exception as e:
        return {"error": f"Process error: {str(e)}"}


@observe()
async def process_github_query(github_url: str, query: str):
    try:
        # Validate GitHub token
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            return {"error": "GITHUB_TOKEN environment variable is required"}
        
        server_params = StdioServerParameters(
            command="docker.exe",
            args=[
                "run",
                "-i",
                "--rm",
                "-e",
                f"GITHUB_PERSONAL_ACCESS_TOKEN={github_token}",
                "ghcr.io/github/github-mcp-server"
            ]
        )
       
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools_result = await client.list_tools()
               
                openai_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("OPENAI_API_VERSION")
                )
                # Enhanced system prompt for deep analysis
                system_prompt = f"""You are a thorough GitHub repository analyst with access to these tools: {', '.join([t.name for t in tools_result.tools])}.

CRITICAL INSTRUCTIONS FOR COMPREHENSIVE ANALYSIS:
1. When exploring directories, you MUST recursively explore ALL subdirectories
2. For each directory found, call the tool again to list its contents
3. For each file found, read its contents to understand the codebase
4. Continue exploring until you've examined the entire repository structure
5. Don't stop after just the root directory - go deep into every folder
6. Make multiple tool calls in sequence to build a complete picture

When asked to summarize a repository:
- First, get the repository structure/tree
- List root directory contents
- For each subdirectory found, recursively list its contents
- Read key configuration files
- Read source code files to understand functionality
- Analyze the overall structure and purpose

Be thorough and methodical. Don't summarize until you've fully explored the repository."""
               
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze {github_url} and answer: {query}"}
                ]
               
                tool_calls = []
               
                for iteration in range(5):
                    try:
                        tools_schema = [{
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                            }
                        } for tool in tools_result.tools]
                       
                        response = openai_client.chat.completions.create(
                            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                            messages=messages,
                            tools=tools_schema,
                            max_tokens=4096,
                            temperature=0.7
                        )
                       
                        message = response.choices[0].message
                       
                        if not message.tool_calls:
                            messages.append({"role": "assistant", "content": message.content or ""})
                            break
                       
                        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        } for tc in message.tool_calls]})
                       
                        for tool_call in message.tool_calls:
                            try:
                                tool_name = tool_call.function.name
                                arguments = json.loads(tool_call.function.arguments)
                               
                                if "url" not in arguments and tool_name in ["analyze_github_repo", "summarize_repository"]:
                                    arguments["url"] = github_url
                               
                                result = await client.call_tool(tool_name, arguments)
                               
                                tool_calls.append({
                                    "tool_name": tool_name,
                                    "arguments": arguments,
                                    "result": result.content
                                })
                               
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(result.content)
                                })
                            except Exception as tool_error:
                                error_msg = str(tool_error)
                                if "content SHA is nil" in error_msg or "403" in error_msg:
                                    error_msg = f"GitHub API Permission Error: {error_msg}. Your token may need 'repo' scope for private repos or 'public_repo' scope for public repos."
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": f"Error: {error_msg}"
                                })
                   
                    except Exception as iter_error:
                        return {"error": f"Iteration {iteration} error: {str(iter_error)}"}
               
                if tool_calls:
                    messages.append({"role": "user", "content": "Based on the tool results above, provide a comprehensive and detailed analysis answering the original query."})
                   
                    final_response = openai_client.chat.completions.create(
                        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                        messages=messages,
                        temperature=0.2
                    )
                    final_answer = final_response.choices[0].message.content
                else:
                    final_answer = next((msg["content"] for msg in reversed(messages) if msg["role"] == "assistant" and msg.get("content")), "No response")
               
                return {
                    "tool_calls": tool_calls,
                    "final_answer": final_answer
                }
                   
    except Exception as e:
        error_msg = str(e)
        if "content SHA is nil" in error_msg:
            error_msg = f"GitHub token permission issue: {error_msg}. Please ensure your GITHUB_TOKEN has proper scopes (repo for private, public_repo for public repositories)."
        return {"error": f"Process error: {error_msg}"}


# Streamlit UI
st.title("Repository Analyzer")
st.write("Analyze GitHub and GitLab repositories using MCP tools")

# Repository type selection
repo_type = st.selectbox("Repository Type", ["GitHub", "GitLab"])

if repo_type == "GitHub":
    with st.expander("GitHub Token Requirements"):
        st.write("**Required token scopes:**")
        st.write("• `public_repo` - for public repositories")
        st.write("• `repo` - for private repositories")
        st.write("• Set your token in the GITHUB_TOKEN environment variable")
    
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    
    if os.getenv('GITHUB_TOKEN'):
        st.success("✅ GitHub token found")
    else:
        st.error("❌ GITHUB_TOKEN not found in environment variables")
        
else:  # GitLab
    with st.expander("GitLab Configuration Requirements"):
        st.write("**Required environment variables:**")
        st.write("• `GITLAB_TOKEN` - GitLab personal access token")
        st.write("")
        st.write("**For full URL (recommended):**")
        st.write("• Just paste the full GitLab URL (e.g., https://gitlab.com/owner/repo.git)")
        st.write("• The server will automatically extract base URL and project path")
        st.write("")
        st.write("**Alternative (if not using full URL):**")
        st.write("• `GITLAB_URL` - GitLab instance URL")
        st.write("• `GITLAB_PROJECT_ID` or `GITLAB_PROJECT_PATH`")
    
    repo_url = st.text_input(
        "GitLab URL", 
        placeholder="https://jitlocker.changepond.com/gitlab-instance-b7d10beb/asisnewcr.git",
        help="Paste your full GitLab repository URL including .git"
    )
    
    if os.getenv('GITLAB_TOKEN'):
        st.success("✅ GitLab token found")
    else:
        st.error("❌ GITLAB_TOKEN not found")
    
    # Show what will be extracted
    if repo_url and repo_url.startswith('http'):
        st.info(f"ℹ️ Will auto-extract configuration from URL: {repo_url}")

user_query = st.text_area("Your Query", placeholder="What would you like to know about this repository?")

if st.button("Analyze"):
    if repo_url and user_query:
        with st.spinner("Processing..."):
            if repo_type == "GitHub":
                result = asyncio.run(process_github_query(repo_url, user_query))
            else:  # GitLab
                result = asyncio.run(process_gitlab_query(repo_url, user_query))
            st.session_state.result = result
    else:
        st.warning(f"Please provide both {repo_type} URL/ID and query")

# Display results
if st.session_state.result:
    result = st.session_state.result
   
    if "error" in result:
        st.error(result["error"])
        st.code(result.get("llm_response", ""))
    else:
        st.success("Analysis Complete!")
       
        with st.expander("Tool Execution Details"):
            for i, tool_call in enumerate(result['tool_calls']):
                st.write(f"**Tool {i+1}:** {tool_call['tool_name']}")
                st.write(f"**Arguments:** {tool_call['arguments']}")
                with st.expander(f"Tool {i+1} Response"):
                    st.code(tool_call['result'], language='json')
       
        st.subheader("Answer")
        st.write(result['final_answer'])