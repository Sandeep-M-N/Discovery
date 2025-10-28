## GitHub MCP Server (stdio)

This project exposes an MCP stdio server that can read, analyze, and summarize a GitHub repository by URL.

### Install

Using pip:

```bash
pip install -e .
```

Or with uv:

```bash
uv sync
```

### Run the server (stdio)

```bash
python server.py
```

The server uses the MCP Python SDK `stdio` transport. Configure your MCP client (e.g., Cursor, Claude Desktop, etc.) to launch this command as a stdio server.

### Tool: analyze_github_repo

- Input: `url` (string) — GitHub URL like `https://github.com/owner/repo` or `https://github.com/owner/repo/tree/branch`
- Optional: `github_token` — GitHub token to increase rate limits or access private repos
- Output: A human-readable summary including counts, top extensions, and sampled file previews

### How it works

- Parses the GitHub URL for `owner`, `repo`, and optional `ref` (branch)
- Uses GitHub Trees API with `recursive=1` to enumerate the whole repo tree without cloning
- Fetches up to 200 files' contents (favoring smaller files) and samples up to 32KB per file
- Produces a concise summary with overall stats and previews

### Notes

- Private repos require `github_token` with appropriate scopes.
- Large repos are summarized with limits to avoid excessive payload size.

### Additional tools

- list_repositories(owner=None, visibility="public", github_token=None)
  - Lists repositories for a specified `owner` (user/org). If `owner` is omitted, lists repos for the authenticated user (requires token). `visibility` can be `all`, `public`, or `private`.

- read_repo_file(owner, repo, path, ref=None, github_token=None)
  - Reads a single file from a repository at `path`, resolves `ref` (branch) if omitted, returns text content (UTF-8), size, and whether truncated.

- analyze_repo_file(owner, repo, path, ref=None, github_token=None)
  - Fetches the file then returns simple per-file analysis metrics (line counts, token histogram, etc.).

- summarize_repository(url, github_token=None)
  - Full detailed repo summary; `analyze_github_repo` is an alias for backward compatibility.

### GitHub token

You can pass a token explicitly to each tool via the `github_token` argument, or set an environment variable `GITHUB_TOKEN` which the server will use automatically when the argument is not provided.

Recommended scopes for private repos: `repo`.

## MCP Client (Azure OpenAI)

Environment variables:

- `AZURE_OPENAI_ENDPOINT` – e.g., `https://your-resource-name.openai.azure.com`
- `AZURE_OPENAI_API_KEY` – Azure OpenAI API key
- `AZURE_OPENAI_DEPLOYMENT` – Chat model deployment name (e.g., `gpt-4o-mini`)
- `GITHUB_TOKEN` – optional, increases rate limits / private repo access

Run client (it will launch the server over stdio automatically if `uv` exists, else uses Python):

```bash
python client.py "Summarize https://github.com/python/cpython and list a few key files"
```

The client:
- Connects to the MCP stdio server
- Lists available tools and exposes them to Azure OpenAI via tool schemas
- Lets the model request tool calls; executes them via MCP; returns results
- Prints final model answer

References:
- MCP Protocol docs: `https://modelcontextprotocol.io`
- MCP Python SDK: `https://github.com/modelcontextprotocol/python-sdk`


