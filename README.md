# Cline MCP Server

Internal workflow orchestration server for agentic task execution.

## Quick Start

```bash
npm install
npm run build
npm start
```

## Docker

```bash
docker build -t cline-mcp .
docker run -p 8080:8080 cline-mcp
```

## Configuration

```json
{
  "database": "sqlite:///data/cline.db",
  "port": 8080,
  "auth_token": "your-secret-token"
}
```

## API Endpoints

### Agent Execution
```http
POST /agent/run
{
  "task": "summarize",
  "input": "today's call notes",
  "tool": "gpt4",
  "context": "internal"
}
```

### Workflow Management
```http
POST /workflow
{
  "workflow_id": "daily-summary",
  "trigger": "n8n",
  "inputs": {...}
}
```

### Memory Operations
```http
GET /memory/{key}
POST /memory
DELETE /memory/{key}
```

### System Execution
```http
POST /exec
{
  "command": "python scripts/process.py",
  "args": ["--input", "data.csv"]
}
```

## MCP Compatibility

Uses unified `/mcp` endpoint for JSON-RPC communication. Supports:
- Tools: agent tasks, workflows, system exec
- Resources: memory access, process logs
- Prompts: task templates, workflow definitions

## License

MIT License