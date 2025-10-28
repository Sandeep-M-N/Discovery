#!/usr/bin/env python3
"""
GitLab MCP Server
Provides tools to interact with GitLab repositories via the Model Context Protocol
"""

import os
import sys
import json
import base64
import re
from typing import Optional, Dict, Any, List
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
from dotenv import load_dotenv
load_dotenv()
# Configuration
GITLAB_URL = os.getenv("GITLAB_URL", "https://gitlab.com")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
GITLAB_PROJECT_ID = os.getenv("GITLAB_PROJECT_ID")
GITLAB_PROJECT_PATH = os.getenv("GITLAB_PROJECT_PATH")

if not GITLAB_TOKEN:
    print("Error: GITLAB_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)

def parse_gitlab_url(url: str) -> tuple[str, str]:
    """Parse GitLab URL to extract base URL and project path"""
    import re
    patterns = [
        r'https?://([^/]+)/(.+?)(?:\.git)?/?$',
        r'https?://([^/]+)/(.+?)/?$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            domain = match.group(1)
            project_path = match.group(2).rstrip('.git')
            base_url = f"https://{domain}"
            return base_url, project_path
    
    raise ValueError(f"Could not parse GitLab URL: {url}")

# If URL is provided, parse it
if os.getenv("GITLAB_REPO_URL"):
    try:
        parsed_url, parsed_path = parse_gitlab_url(os.getenv("GITLAB_REPO_URL"))
        GITLAB_URL = parsed_url
        GITLAB_PROJECT_PATH = parsed_path
    except ValueError as e:
        print(f"Error parsing GitLab URL: {e}", file=sys.stderr)
        sys.exit(1)

if not GITLAB_PROJECT_ID and not GITLAB_PROJECT_PATH and not os.getenv("GITLAB_REPO_URL"):
    print("Error: Either GITLAB_PROJECT_ID, GITLAB_PROJECT_PATH, or GITLAB_REPO_URL environment variable is required", file=sys.stderr)
    sys.exit(1)


class GitLabAPI:
    """GitLab API client"""
    
    def __init__(self, url: str, token: str, project_id: Optional[str] = None, project_path: Optional[str] = None):
        self.base_url = f"{url}/api/v4"
        self.token = token
        self.headers = {
            "PRIVATE-TOKEN": token,
            "Content-Type": "application/json"
        }
        
        # Prefer project ID over path for reliability
        if project_id:
            self.project_identifier = project_id
        elif project_path:
            from urllib.parse import quote
            self.project_identifier = quote(project_path, safe='')
        else:
            raise ValueError("Either project_id or project_path must be provided")
    
    async def request(self, endpoint: str) -> Any:
        """Make API request to GitLab"""
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
    
    async def list_repository(self, path: str = "", ref: str = "main") -> List[Dict]:
        """List repository tree"""
        endpoint = f"/projects/{self.project_identifier}/repository/tree"
        params = f"?ref={ref}&path={path}" if path else f"?ref={ref}"
        return await self.request(f"{endpoint}{params}")
    
    async def get_raw_file(self, file_path: str, ref: str = "main") -> str:
        """Get raw file content"""
        from urllib.parse import quote
        encoded_path = quote(file_path, safe='')
        endpoint = f"/projects/{self.project_identifier}/repository/files/{encoded_path}/raw?ref={ref}"
        
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            return response.text


class FileAnalyzer:
    """Analyze file contents"""
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get file extension"""
        parts = file_path.split('.')
        return f".{parts[-1]}" if len(parts) > 1 else ""
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Determine file type from extension"""
        ext = FileAnalyzer.get_file_extension(file_path).lower()
        type_map = {
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "React JSX",
            ".tsx": "React TSX",
            ".py": "Python",
            ".java": "Java",
            ".c": "C",
            ".cpp": "C++",
            ".go": "Go",
            ".rs": "Rust",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".xml": "XML",
            ".md": "Markdown",
            ".txt": "Text",
            ".html": "HTML",
            ".css": "CSS",
            ".sql": "SQL",
            ".rb": "Ruby",
            ".php": "PHP",
            ".sh": "Shell",
        }
        return type_map.get(ext, "Unknown")
    
    @staticmethod
    def analyze_code(content: str) -> Dict[str, Any]:
        """Analyze code files"""
        lines = content.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # Detect comments
        comment_patterns = [r'^\s*#', r'^\s*//', r'^\s*/\*', r'^\s*\*']
        comment_lines = [l for l in lines if any(re.match(p, l) for p in comment_patterns)]
        
        # Detect functions
        function_patterns = [
            r'\bfunction\s+\w+',
            r'\bdef\s+\w+',
            r'\bfunc\s+\w+',
            r'\bfn\s+\w+',
            r'\bsub\s+\w+',
        ]
        functions = sum(len(re.findall(p, content)) for p in function_patterns)
        
        # Detect classes
        classes = len(re.findall(r'\bclass\s+\w+', content))
        
        # Detect imports
        import_patterns = [
            r'\bimport\s+',
            r'\bfrom\s+\w+\s+import',
            r'\brequire\(',
            r'\buse\s+',
            r'\binclude\s+',
        ]
        imports = sum(len(re.findall(p, content)) for p in import_patterns)
        
        return {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "comment_lines": len(comment_lines),
            "blank_lines": len(lines) - len(non_empty_lines),
            "functions_detected": functions,
            "classes_detected": classes,
            "imports_detected": imports,
        }
    
    @staticmethod
    def analyze_json(content: str) -> Dict[str, Any]:
        """Analyze JSON files"""
        try:
            data = json.loads(content)
            
            def get_depth(obj, depth=0):
                if not isinstance(obj, (dict, list)):
                    return depth
                if isinstance(obj, dict):
                    return max([get_depth(v, depth + 1) for v in obj.values()], default=depth)
                if isinstance(obj, list):
                    return max([get_depth(item, depth + 1) for item in obj], default=depth)
            
            return {
                "valid": True,
                "depth": get_depth(data),
                "keys": len(data) if isinstance(data, dict) else None,
                "items": len(data) if isinstance(data, list) else None,
            }
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    @staticmethod
    def analyze_text(content: str) -> Dict[str, Any]:
        """Analyze text files"""
        words = [w for w in re.split(r'\s+', content) if w]
        paragraphs = [p for p in re.split(r'\n\n+', content) if p.strip()]
        
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        return {
            "word_count": len(words),
            "paragraph_count": len(paragraphs),
            "average_word_length": round(avg_word_length, 2),
        }
    
    @staticmethod
    def analyze_file(content: str, file_path: str) -> Dict[str, Any]:
        """Comprehensive file analysis"""
        analysis = {
            "file_path": file_path,
            "size_bytes": len(content.encode('utf-8')),
            "line_count": len(content.split('\n')),
            "character_count": len(content),
            "file_type": FileAnalyzer.get_file_type(file_path),
            "extension": FileAnalyzer.get_file_extension(file_path),
        }
        
        ext = analysis["extension"].lower()
        
        # Code analysis
        if ext in [".js", ".ts", ".jsx", ".tsx", ".py", ".java", ".c", ".cpp", 
                   ".go", ".rs", ".rb", ".php", ".sh"]:
            analysis["code_analysis"] = FileAnalyzer.analyze_code(content)
        
        # JSON analysis
        if ext == ".json":
            analysis["structure_analysis"] = FileAnalyzer.analyze_json(content)
        
        # YAML/XML detection
        if ext in [".yaml", ".yml"]:
            analysis["structure_analysis"] = {
                "type": "YAML",
                "note": "YAML file detected"
            }
        
        if ext == ".xml":
            analysis["structure_analysis"] = {
                "type": "XML",
                "valid": "<?xml" in content or "<" in content
            }
        
        # Text analysis
        if ext in [".md", ".txt", ".rst"]:
            analysis["text_analysis"] = FileAnalyzer.analyze_text(content)
        
        return analysis


# Initialize GitLab API client
print(f"Initializing GitLab API with:")
print(f"  URL: {GITLAB_URL}")
print(f"  Project ID: {GITLAB_PROJECT_ID}")
print(f"  Project Path: {GITLAB_PROJECT_PATH}")

gitlab = GitLabAPI(
    GITLAB_URL, 
    GITLAB_TOKEN, 
    project_id=GITLAB_PROJECT_ID,
    project_path=GITLAB_PROJECT_PATH
)
analyzer = FileAnalyzer()

# Create MCP server
app = Server("gitlab-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="list_directories",
            description="List files and directories in a GitLab repository path. Returns information about each item including name, type, path, and mode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list (empty string for root)",
                        "default": "",
                    },
                    "ref": {
                        "type": "string",
                        "description": "The branch, tag, or commit SHA to use",
                        "default": "main",
                    },
                },
            },
        ),
        Tool(
            name="read_file",
            description="Read the contents of a file from the GitLab repository. Returns the raw file content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The full path to the file in the repository",
                    },
                    "ref": {
                        "type": "string",
                        "description": "The branch, tag, or commit SHA to use",
                        "default": "main",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="analyze_file",
            description="Analyze a file from the GitLab repository. Provides detailed analysis including line count, file type, size, and language-specific metrics (code complexity, JSON structure, text statistics, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The full path to the file in the repository",
                    },
                    "ref": {
                        "type": "string",
                        "description": "The branch, tag, or commit SHA to use",
                        "default": "main",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        if name == "list_directories":
            path = arguments.get("path", "")
            ref = arguments.get("ref", "main")
            
            items = await gitlab.list_repository(path, ref)
            formatted = [
                {
                    "name": item["name"],
                    "path": item["path"],
                    "type": item["type"],
                    "mode": item.get("mode", ""),
                }
                for item in items
            ]
            
            return [TextContent(
                type="text",
                text=json.dumps(formatted, indent=2)
            )]
        
        elif name == "read_file":
            file_path = arguments.get("file_path")
            ref = arguments.get("ref", "main")
            
            if not file_path:
                raise ValueError("file_path is required")
            
            content = await gitlab.get_raw_file(file_path, ref)
            
            return [TextContent(
                type="text",
                text=content
            )]
        
        elif name == "analyze_file":
            file_path = arguments.get("file_path")
            ref = arguments.get("ref", "main")
            
            if not file_path:
                raise ValueError("file_path is required")
            
            content = await gitlab.get_raw_file(file_path, ref)
            analysis = analyzer.analyze_file(content, file_path)
            
            return [TextContent(
                type="text",
                text=json.dumps(analysis, indent=2)
            )]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Main entry point"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())