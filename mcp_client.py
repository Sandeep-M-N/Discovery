import asyncio
import json
import os
import sys
from typing import Any, Dict, List
from openai import AzureOpenAI
from dotenv import load_dotenv

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
load_dotenv()

server_params = StdioServerParameters(
    command="uv",
    args=["run", "server.py"],
    env={
        "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")
    }


)
def llm_client(user_prompt: str): 
    openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )
    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=2024,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def get_prompt_to_identify_tool_and_arguments(query, tools_result):
    tool_description = "\n".join([f"- {t.name}: {t.description}\n  Input: {t.inputSchema}" for t in tools_result.tools])
    return (f"Choose the appropriate tool for: {query}\n\n"
            f"Available tools:\n{tool_description}\n\n"
            "Return JSON with 'tool_name' and 'arguments' fields. "
            "For GitHub URLs, use 'analyze_github_repo' or 'summarize_repository' tool with 'url' argument.")


async def run(query: str):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            tools_result = await client.list_tools()
            
            # Get tool selection from LLM
            prompt = get_prompt_to_identify_tool_and_arguments(query, tools_result)
            llm_response = llm_client(prompt)
            
            try:
                tool_call = json.loads(llm_response)
                tool_name = tool_call.get("tool_name")
                arguments = tool_call.get("arguments", {})
                
                print(f"\n=== TOOL EXECUTION ===")
                print(f"Tool Called: {tool_name}")
                print(f"Arguments: {arguments}")
                
                # Execute the MCP tool
                result = await client.call_tool(tool_name, arguments)
                print(f"\n=== TOOL RESPONSE ===")
                print(f"Tool Result: {result.content}")
                print(f"\n=== END TOOL RESPONSE ===")
                
                # Get final answer from LLM using tool results
                final_prompt = f"Based on this data: {result.content}\n\nAnswer the user's question: {query}"
                final_answer = llm_client(final_prompt)
                print(f"Final Answer: {final_answer}")
                
            except json.JSONDecodeError:
                print(f"LLM Response (no tool needed): {llm_response}")

if __name__ == "__main__":
    query = "show me the dependency mapping of this github repository https://github.com/vasanthravi7/wvc-social-media"
    asyncio.run(run(query))



# Load environment variables from .env file
