from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool
def test_tool(query: str) -> str:
    """A simple test tool"""
    return f"Test result for: {query}"

def check_tool_support(model_name: str) -> bool:
    """Check if a model supports tool calling"""
    try:
        model = ChatOllama(model=model_name)
        model_with_tools = model.bind_tools([test_tool])
        response = model_with_tools.invoke("test")
        return True
    except Exception as e:
        if "does not support tools" in str(e):
            return False
        raise e

# Test models
models_to_test = ["deepseek-r1", "llama3.1", "qwen3:latest", "mistral"]

for model in models_to_test:
    try:
        supports_tools = check_tool_support(model)
        print(f"{model}: {'✓ Supports tools' if supports_tools else '✗ No tool support'}")
    except Exception as e:
        print(f"{model}: Error - {e}")