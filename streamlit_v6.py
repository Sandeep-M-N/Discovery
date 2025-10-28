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
if 'pending_tools' not in st.session_state:
    st.session_state.pending_tools = []
if 'approved_tools' not in st.session_state:
    st.session_state.approved_tools = {}
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = 'manual'  # 'manual' or 'auto'
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0


def display_tool_approval_ui():
    """Display pending tools for user approval"""
    if not st.session_state.pending_tools:
        return False
    
    st.warning(f"🔧 {len(st.session_state.pending_tools)} tool(s) requesting execution")
    
    for idx, tool_info in enumerate(st.session_state.pending_tools):
        with st.expander(f"Tool {idx + 1}: {tool_info['name']}", expanded=True):
            st.write(f"**Description:** {tool_info.get('description', 'No description')}")
            st.write("**Arguments:**")
            st.json(tool_info['arguments'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ Approve", key=f"approve_{idx}"):
                    st.session_state.approved_tools[idx] = True
                    st.rerun()
            with col2:
                if st.button(f"❌ Reject", key=f"reject_{idx}"):
                    st.session_state.approved_tools[idx] = False
                    st.rerun()
            with col3:
                if st.button(f"⏭️ Skip", key=f"skip_{idx}"):
                    st.session_state.approved_tools[idx] = None
                    st.rerun()
    
    # Check if all tools have been reviewed
    all_reviewed = all(idx in st.session_state.approved_tools for idx in range(len(st.session_state.pending_tools)))
    
    if all_reviewed:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Execute Approved Tools"):
                return True
        with col2:
            if st.button("🔄 Reset Approvals"):
                st.session_state.approved_tools = {}
                st.rerun()
    
    return False


@observe()
async def process_gitlab_query(gitlab_url: str, query: str, approval_callback=None):
    try:
        gitlab_token = os.getenv('GITLAB_TOKEN')
        
        if not gitlab_token:
            return {"error": "GITLAB_TOKEN environment variable is required"}
        
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
                
                system_prompt = f"""You are a thorough GitLab repository analyst with access to these tools: {', '.join([t.name for t in tools_result.tools])}.

ABSOLUTE RULES - NO EXCEPTIONS:

1. You MUST call list_directories on EVERY SINGLE directory path you see
2. When you see a subdirectory like "langgraph_workflow/", you MUST immediately call list_directories("path/to/langgraph_workflow")
3. NEVER describe a directory without first calling list_directories on it
4. NEVER write a summary until you've explored EVERY directory path mentioned
5. If you list a directory and see subdirectories, your VERY NEXT tool call must explore those subdirectories
6. Inside those subdirectories, you may find another directories,you MUST immediately explore it with the appropriate tool
7. You have 20 iterations - use ALL of them to explore deeply
"""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze the GitLab repository and answer: {query}"}
                ]
                
                tool_calls = []
                directories_explored = set()
                directories_found = set()
                verification_prompted = False
                
                for iteration in range(20):  # Increased from 10 to 20 for deep repositories
                    st.session_state.current_iteration = iteration + 1
                    
                    # Show progress in sidebar
                    if iteration > 0:
                        explored_count = len(directories_explored)
                        found_count = len(directories_found)
                        st.sidebar.metric("Directories Found", found_count)
                        st.sidebar.metric("Directories Explored", explored_count)
                        if found_count > 0:
                            progress = explored_count / found_count
                            st.sidebar.progress(min(progress, 1.0), f"Exploration: {int(progress * 100)}%")
                    
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
                        
                        # Prepare tool calls for approval
                        pending_tools = []
                        for tc in message.tool_calls:
                            tool_info = {
                                'id': tc.id,
                                'name': tc.function.name,
                                'arguments': json.loads(tc.function.arguments),
                                'description': next((t.description for t in tools_result.tools if t.name == tc.function.name), "")
                            }
                            pending_tools.append(tool_info)
                        
                        # Request approval if in manual mode
                        if approval_callback and st.session_state.processing_mode == 'manual':
                            st.session_state.pending_tools = pending_tools
                            approved = await approval_callback()
                            
                            if not approved:
                                return {"error": "Tool execution cancelled by user"}
                        else:
                            # Auto-approve all tools
                            st.session_state.approved_tools = {i: True for i in range(len(pending_tools))}
                        
                        # Add assistant message with tool calls
                        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        } for tc in message.tool_calls]})
                        
                        # Execute approved tools
                        for idx, tool_call in enumerate(message.tool_calls):
                            if st.session_state.approved_tools.get(idx) == True:
                                try:
                                    tool_name = tool_call.function.name
                                    arguments = json.loads(tool_call.function.arguments)
                                    
                                    # Track directory exploration
                                    if tool_name == "gitlab-mcp-server:list_directories":
                                        path = arguments.get('path', '')
                                        directories_explored.add(path)
                                    
                                    result = await client.call_tool(tool_name, arguments)
                                    
                                    # Track directories found in results
                                    if tool_name == "gitlab-mcp-server:list_directories":
                                        try:
                                            result_content = str(result.content)
                                            # More comprehensive directory parsing
                                            import re
                                            
                                            # Method 1: Parse JSON-like structure for paths with type "tree"
                                            path_pattern = r'"path":\s*"([^"]+)"[^}]*?"type":\s*"tree"'
                                            found_dirs = re.findall(path_pattern, result_content)
                                            directories_found.update(found_dirs)
                                            
                                            # Method 2: Look for directory paths ending with /
                                            dir_pattern = r'([a-zA-Z0-9_\-./]+/)(?=[\s,\]\}]|$)'
                                            found_dirs2 = re.findall(dir_pattern, result_content)
                                            directories_found.update([d.rstrip('/') for d in found_dirs2])
                                            
                                            # Method 3: Parse TextContent objects
                                            text_content_pattern = r'text="([^"]*)"'
                                            text_contents = re.findall(text_content_pattern, result_content)
                                            for text in text_contents:
                                                if '/' in text and not text.startswith('http'):
                                                    directories_found.add(text.rstrip('/'))
                                        except:
                                            pass
                                    
                                    tool_calls.append({
                                        "tool_name": tool_name,
                                        "arguments": arguments,
                                        "result": result.content,
                                        "approved": True
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
                            else:
                                # Tool was rejected or skipped
                                tool_calls.append({
                                    "tool_name": tool_call.function.name,
                                    "arguments": json.loads(tool_call.function.arguments),
                                    "result": "Rejected by user",
                                    "approved": False
                                })
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": "Tool execution rejected by user"
                                })
                        
                        # Clear pending tools after execution
                        st.session_state.pending_tools = []
                        st.session_state.approved_tools = {}
                        
                        # Check if there are unexplored directories and prompt verification
                        unexplored = directories_found - directories_explored
                        if unexplored and not verification_prompted and iteration >= 1:  # Changed from > 2 to >= 1
                            verification_prompted = True
                            unexplored_list = list(unexplored)[:10]  # Show up to 10
                            dir_list = '\n'.join([f"- {d}" for d in unexplored_list])
                            messages.append({
                                "role": "user",
                                "content": f"""STOP! You have NOT explored these directories yet:
{dir_list}

You MUST call list_directories on EACH of these paths RIGHT NOW. Do not write any summary or analysis until you've explored every single one of these directories. Start immediately with the first unexplored directory."""
                            })
                            verification_prompted = False  # Allow multiple verification prompts
                    
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
async def process_github_query(github_url: str, query: str, approval_callback=None):
    try:
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
                
                system_prompt = f"""You are a thorough GitHub repository analyst with access to these tools: {', '.join([t.name for t in tools_result.tools])}.

ABSOLUTE RULES - NO EXCEPTIONS:

1. You MUST explore EVERY SINGLE directory path you encounter
2. When you see a subdirectory, you MUST immediately explore it with the appropriate tool
3. NEVER describe a directory without first exploring its contents
4. NEVER write a summary until you've explored EVERY directory path mentioned
5. If you discover subdirectories, your VERY NEXT tool calls must explore those subdirectories
6. Inside those subdirectories, you may find another directories,you MUST immediately explore it with the appropriate tool
7. You have 20 iterations - use ALL of them to explore deeply

"""
               
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze {github_url} and answer: {query}"}
                ]
               
                tool_calls = []
                directories_explored = set()
                directories_found = set()
                verification_prompted = False
               
                for iteration in range(20):  # Increased from 10 to 20 for deep repositories
                    st.session_state.current_iteration = iteration + 1
                    
                    # Show progress in sidebar  
                    if iteration > 0:
                        explored_count = len(directories_explored)
                        found_count = len(directories_found)
                        st.sidebar.metric("Directories Found", found_count)
                        st.sidebar.metric("Directories Explored", explored_count)
                        if found_count > 0:
                            progress = explored_count / found_count
                            st.sidebar.progress(min(progress, 1.0), f"Exploration: {int(progress * 100)}%")
                    
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
                       
                        # Prepare tool calls for approval
                        pending_tools = []
                        for tc in message.tool_calls:
                            tool_info = {
                                'id': tc.id,
                                'name': tc.function.name,
                                'arguments': json.loads(tc.function.arguments),
                                'description': next((t.description for t in tools_result.tools if t.name == tc.function.name), "")
                            }
                            pending_tools.append(tool_info)
                        
                        # Request approval if in manual mode
                        if approval_callback and st.session_state.processing_mode == 'manual':
                            st.session_state.pending_tools = pending_tools
                            approved = await approval_callback()
                            
                            if not approved:
                                return {"error": "Tool execution cancelled by user"}
                        else:
                            # Auto-approve all tools
                            st.session_state.approved_tools = {i: True for i in range(len(pending_tools))}
                        
                        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        } for tc in message.tool_calls]})
                       
                        for idx, tool_call in enumerate(message.tool_calls):
                            if st.session_state.approved_tools.get(idx) == True:
                                try:
                                    tool_name = tool_call.function.name
                                    arguments = json.loads(tool_call.function.arguments)
                                   
                                    if "url" not in arguments and tool_name in ["analyze_github_repo", "summarize_repository"]:
                                        arguments["url"] = github_url
                                    
                                    # Track directory exploration for GitHub
                                    if 'path' in arguments and tool_name in ["get_file_contents", "list_directory_contents"]:
                                        path = arguments.get('path', '')
                                        if path:
                                            directories_explored.add(path)
                                   
                                    result = await client.call_tool(tool_name, arguments)
                                    
                                    # Track directories found in GitHub results
                                    try:
                                        result_content = str(result.content)
                                        if 'directory' in result_content.lower() or 'folder' in result_content.lower():
                                            import re
                                            # Try to find directory paths in the result
                                            path_pattern = r'([a-zA-Z0-9_\-/]+/)(?=\s|$|,)'
                                            found_dirs = re.findall(path_pattern, result_content)
                                            directories_found.update(found_dirs)
                                    except:
                                        pass
                                   
                                    tool_calls.append({
                                        "tool_name": tool_name,
                                        "arguments": arguments,
                                        "result": result.content,
                                        "approved": True
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
                            else:
                                tool_calls.append({
                                    "tool_name": tool_call.function.name,
                                    "arguments": json.loads(tool_call.function.arguments),
                                    "result": "Rejected by user",
                                    "approved": False
                                })
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": "Tool execution rejected by user"
                                })
                        
                        st.session_state.pending_tools = []
                        st.session_state.approved_tools = {}
                        
                        # Check if there are unexplored directories and prompt verification
                        unexplored = directories_found - directories_explored
                        if unexplored and iteration >= 1:  # Check early and often
                            unexplored_list = list(unexplored)[:10]
                            dir_list = '\n'.join([f"- {d}" for d in unexplored_list])
                            messages.append({
                                "role": "user",
                                "content": f"""STOP! You have NOT explored these directories yet:
{dir_list}

You MUST explore EACH of these paths RIGHT NOW using the appropriate tools. Do not write any summary or analysis until you've explored every single one. Start immediately with the first unexplored directory."""
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
st.title("🔍 Repository Analyzer with Tool Approval")
st.write("Analyze GitHub and GitLab repositories using MCP tools with user approval")

# Approval mode selection in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    approval_mode = st.radio(
        "Tool Approval Mode",
        ["Manual Approval", "Auto-approve All"],
        help="Choose whether to manually approve each tool execution or auto-approve all"
    )
    st.session_state.processing_mode = 'manual' if approval_mode == "Manual Approval" else 'auto'
    
    if st.session_state.processing_mode == 'manual':
        st.info("🔒 Each tool will require your approval before execution")
    else:
        st.warning("⚡ All tools will execute automatically")
    
    if st.session_state.current_iteration > 0:
        st.metric("Current Iteration", st.session_state.current_iteration)

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
    
    repo_url = st.text_input(
        "GitLab URL", 
        placeholder="https://jitlocker.changepond.com/gitlab-instance-b7d10beb/asisnewcr.git",
        help="Paste your full GitLab repository URL including .git"
    )
    
    if os.getenv('GITLAB_TOKEN'):
        st.success("✅ GitLab token found")
    else:
        st.error("❌ GITLAB_TOKEN not found")

user_query = st.text_area("Your Query", placeholder="What would you like to know about this repository?")

# Tool approval section
if st.session_state.pending_tools:
    st.divider()
    st.header("🔧 Tool Approval Required")
    ready_to_execute = display_tool_approval_ui()
    
    if ready_to_execute:
        st.success("Continuing with approved tools...")
        st.rerun()

if st.button("🚀 Analyze", disabled=bool(st.session_state.pending_tools)):
    if repo_url and user_query:
        # Reset state
        st.session_state.pending_tools = []
        st.session_state.approved_tools = {}
        st.session_state.current_iteration = 0
        
        with st.spinner("Processing..."):
            # This is a simplified approach - in production you'd need a more sophisticated
            # callback mechanism since Streamlit doesn't support async callbacks natively
            async def dummy_callback():
                return True
            
            if repo_type == "GitHub":
                result = asyncio.run(process_github_query(repo_url, user_query, None if st.session_state.processing_mode == 'auto' else dummy_callback))
            else:
                result = asyncio.run(process_gitlab_query(repo_url, user_query, None if st.session_state.processing_mode == 'auto' else dummy_callback))
            
            st.session_state.result = result
    else:
        st.warning(f"Please provide both {repo_type} URL and query")

# Display results
if st.session_state.result and not st.session_state.pending_tools:
    result = st.session_state.result
   
    if "error" in result:
        st.error(result["error"])
    else:
        st.success("✅ Analysis Complete!")
       
        with st.expander("📋 Tool Execution Details", expanded=False):
            for i, tool_call in enumerate(result['tool_calls']):
                status = "✅ Approved" if tool_call.get('approved', True) else "❌ Rejected"
                st.write(f"**Tool {i+1}:** {tool_call['tool_name']} {status}")
                st.write(f"**Arguments:** {tool_call['arguments']}")
                with st.expander(f"Tool {i+1} Response"):
                    st.code(tool_call['result'], language='json')
       
        st.subheader("📝 Final Answer")
        st.markdown(result['final_answer'])