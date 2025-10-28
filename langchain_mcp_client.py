from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint
import asyncio
import os 
from dotenv import load_dotenv
load_dotenv()

model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7
)
# print(model.invoke("hi"))
# model=ChatGroq(
#    model="openai/gpt-oss-120b",
#    temperature=0.6,
#    api_key=os.getenv("GROQ_API_KEY")
# )

server_params = StdioServerParameters(
    command="docker.exe",
    args=[
                "run",
                "-i",
                "--rm",
                "-e",
                f"GITHUB_PERSONAL_ACCESS_TOKEN={os.getenv("GITHUB_TOKEN")}",
                "ghcr.io/github/github-mcp-server"
            ]
)
async def run_agent():
    async with stdio_client(server_params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            print("MCP session initialized..")
            tools= await load_mcp_tools(session)
            print(f"Loaded Tools: {[tool.name for tool in tools]}")
            agent=create_agent(model,tools)
            print("Agent created..")
            response= await agent.ainvoke(
                {"messages":[
                    {
                        "role":"user",
                        "content": "show me the dependency mapping of this url https://github.com/Sandeep-M-N/Acumen"

                    }
                ]
                 }
            )
            print("agent invocation completed..")
            return response["messages"][-1].content
        
# Standard Python entry point check

if __name__ == "__main__":
  # Run the asynchronous run_agent function and wait for the result

 print("Starting MCP Client...")

 result = asyncio.run(run_agent())

 print("\nAgent Final Response:")

 print(result)