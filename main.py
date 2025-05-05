#!/usr/bin/env python3
"""
Cline MCP Server - Internal workflow orchestration for agentic tasks
"""
import asyncio
import json
import sqlite3
import subprocess
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Initialize FastAPI app with MCP support
app = FastAPI(title="Cline MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_PATH = Path(__file__).parent / "data" / "cline.db"
DB_PATH.parent.mkdir(exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            key TEXT PRIMARY KEY,
            value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            name TEXT,
            config TEXT,
            status TEXT DEFAULT 'active'
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Models
class AgentTask(BaseModel):
    task: str
    input: str
    tool: str = "gpt4"
    context: str = "internal"

class WorkflowRequest(BaseModel):
    workflow_id: str
    trigger: str = "n8n"
    inputs: Dict[str, Any] = {}

class ExecCommand(BaseModel):
    command: str
    args: List[str] = []
    working_dir: Optional[str] = None

class MemoryItem(BaseModel):
    key: str
    value: Any

# Memory operations
def get_memory(key: str) -> Optional[Any]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM memory WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return json.loads(result[0]) if result else None

def set_memory(key: str, value: Any):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)",
        (key, json.dumps(value))
    )
    conn.commit()
    conn.close()

# Workspace endpoints
@app.post("/agent/run", response_model=Dict[str, Any])
async def run_agent(task: AgentTask):
    """Execute an agent task with the specified tool"""
    task_id = str(uuid.uuid4())
    
    # Store task in memory for tracking
    set_memory(f"task:{task_id}", {
        "status": "processing",
        "task": task.task,
        "input": task.input,
        "tool": task.tool,
        "context": task.context
    })
    
    # Simulate tool execution (replace with actual tool integration)
    result = f"Task '{task.task}' executed using {task.tool}"
    
    # Update task status
    set_memory(f"task:{task_id}", {
        "status": "completed",
        "result": result,
        "task": task.task
    })
    
    return {
        "task_id": task_id,
        "status": "completed",
        "result": result
    }

@app.post("/workflow", response_model=Dict[str, Any])
async def execute_workflow(workflow: WorkflowRequest):
    """Execute a workflow triggered from n8n or other sources"""
    workflow_id = workflow.workflow_id
    
    # Load workflow config
    workflow_config = get_memory(f"workflow:{workflow_id}")
    if not workflow_config:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    # Execute workflow steps
    execution_id = str(uuid.uuid4())
    results = []
    
    # Store execution details
    set_memory(f"execution:{execution_id}", {
        "workflow_id": workflow_id,
        "trigger": workflow.trigger,
        "inputs": workflow.inputs,
        "status": "running",
        "results": results
    })
    
    # Simulate workflow execution
    for i in range(3):  # Example workflow with 3 steps
        step_result = f"Step {i+1} completed"
        results.append(step_result)
    
    # Update execution status
    set_memory(f"execution:{execution_id}", {
        "workflow_id": workflow_id,
        "status": "completed",
        "results": results
    })
    
    return {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "status": "completed",
        "results": results
    }

@app.get("/memory/{key}")
async def get_memory_endpoint(key: str):
    """Retrieve memory value by key"""
    value = get_memory(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Key {key} not found")
    return {"key": key, "value": value}

@app.post("/memory")
async def set_memory_endpoint(item: MemoryItem):
    """Store memory value with key"""
    set_memory(item.key, item.value)
    return {"status": "success", "key": item.key}

@app.delete("/memory/{key}")
async def delete_memory_endpoint(key: str):
    """Delete memory value by key"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory WHERE key = ?", (key,))
    conn.commit()
    conn.close()
    return {"status": "success", "key": key}

@app.post("/exec", response_model=Dict[str, Any])
async def execute_command(cmd: ExecCommand):
    """Execute a system command or script"""
    try:
        # Build command with arguments
        full_command = [cmd.command] + cmd.args
        
        # Execute command
        process = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            cwd=cmd.working_dir,
            timeout=30  # 30 second timeout
        )
        
        return {
            "status": "success",
            "exit_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": "Command execution timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# MCP Protocol Endpoints
@app.all("/mcp")
async def mcp_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    """Unified MCP endpoint for JSON-RPC communication"""
    if request.method == "GET":
        # Handle SSE connection
        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    if request.method == "POST":
        # Handle JSON-RPC requests
        try:
            body = await request.json()
            method = body.get("method")
            params = body.get("params", {})
            
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": "run_agent_task",
                                "description": "Execute an agent task",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "task": {"type": "string"},
                                        "input": {"type": "string"},
                                        "tool": {"type": "string"}
                                    }
                                }
                            },
                            {
                                "name": "execute_workflow",
                                "description": "Execute a workflow",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "workflow_id": {"type": "string"},
                                        "inputs": {"type": "object"}
                                    }
                                }
                            },
                            {
                                "name": "run_command",
                                "description": "Execute a system command",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string"},
                                        "args": {"type": "array"}
                                    }
                                }
                            }
                        ]
                    }
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if tool_name == "run_agent_task":
                    task = AgentTask(**arguments)
                    result = await run_agent(task)
                    return {
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
                    }
                
            return {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": "Method not found"}}
            
        except Exception as e:
            return {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32000, "message": str(e)}}

async def generate_sse():
    """Generate Server-Sent Events for SSE transport"""
    while True:
        # Send periodic keep-alive events
        yield f"event: ping\ndata: {json.dumps({'timestamp': asyncio.get_event_loop().time()})}\n\n"
        await asyncio.sleep(30)

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker containers"""
    try:
        # Test database connection
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
