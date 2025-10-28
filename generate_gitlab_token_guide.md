# GitLab Personal Access Token Setup

## Your Issue
Your current token is getting a 302 redirect to sign_in, which means:
- Token is invalid/expired
- Token doesn't have proper permissions
- Token format is incorrect

## How to Create/Update GitLab Token

### Step 1: Access Token Settings
1. Go to your GitLab instance: `https://jitlocker.changepond.com/gitlab-instance-b7d10beb`
2. Sign in to your account
3. Click your avatar (top right) → **Preferences**
4. In left sidebar, click **Access Tokens**

### Step 2: Create New Token
1. Click **Add new token**
2. **Name**: `MCP Repository Analyzer`
3. **Expiration date**: Set future date (or leave blank for no expiration)
4. **Select scopes** (check these boxes):
   - ✅ `api` - Full API access
   - ✅ `read_api` - Read API access
   - ✅ `read_repository` - Read repository files
   - ✅ `read_user` - Read user information

### Step 3: Required Scopes for Your Use Case
For repository analysis, you need:
- `read_api` - To access GitLab API
- `read_repository` - To read repository files and structure
- `read_user` - To verify token validity

### Step 4: Copy and Update Token
1. Click **Create personal access token**
2. **IMPORTANT**: Copy the token immediately (it won't be shown again)
3. Update your `.env` file:
   ```
   GITLAB_TOKEN="glpat-your-new-token-here"
   ```

## Token Format
GitLab tokens should start with `glpat-` followed by alphanumeric characters.

## Test Your New Token
Run the token checker again after updating:
```bash
python check_gitlab_token.py
```

## Common Issues
- **401 Unauthorized**: Token invalid or insufficient permissions
- **403 Forbidden**: Token valid but lacks required scope
- **404 Not Found**: Project doesn't exist or no access
- **302 Redirect**: Token expired or invalid (your current issue)