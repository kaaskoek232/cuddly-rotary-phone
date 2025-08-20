# removed unused import from nt
import sys
import json
import logging
import time
import requests
import hashlib
import re
import argparse
import asyncio
import socket
import subprocess
import concurrent.futures
import pypdf
import feedparser
import ast
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Any
from bs4 import BeautifulSoup
import fastmcp
from fastmcp import FastMCP, Client, Context
from fastmcp.tools import Tool, FunctionTool
from fastmcp.tools.tool_transform import ArgTransform
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import os

# Pydantic Models for FastAPI
class SearchRequest(BaseModel):
    query: str = Field(..., description="The search topic or question")
    sources: List[str] = Field(default=["brave", "context7", "arxiv", "github", "memory"], description="List of sources to search")
    synthesize: bool = Field(default=True, description="Whether to synthesize results")
    max_results: int = Field(default=5, description="Maximum results per source")

class MemoryRequest(BaseModel):
    action: str = Field(..., description="Action: save, retrieve, or search")
    key: Optional[str] = Field(None, description="Memory key")
    value: Optional[str] = Field(None, description="Memory value")
    query: Optional[str] = Field(None, description="Search query")
    category: Optional[str] = Field("", description="Memory category")
    tags: List[str] = Field(default=[], description="Memory tags")

class CodebaseAnalysisRequest(BaseModel):
    project_root: str = Field(..., description="Root directory of the codebase")
    pattern: str = Field("", description="Optional pattern filter")
    max_files: int = Field(50, description="Maximum files to analyze")

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: float
    checks: Dict[str, Any]
    summary: Dict[str, Any]
    issues: List[str]

# Global Constants (can be overridden by args in main_async)
DEBUG_MODE = True
OLLAMA_API_URL = "http://localhost:11434"
MEMORY_JSON_PATH = Path.home() / '.agentic_memory.json'
MAX_MEMORY_ENTRIES = 1000
LLM_TIMEOUT = 500
CACHE_TTL_SECONDS = 3600
OLLAMA_MODEL_NAME = "Gemma-2:latest"
BRAVE_API_KEY = "BSAA7gKWCOziPdagJjAfYMOIfnc3sFj"
CONTEXT7_MCP_SERVER_URL = "https://mcp.context7.com/mcp"
ARXIV_API_URL = "http://export.arxiv.org/api/query"
GITHUB_API_URL = "https://api.github.com/search/repositories"
API_KEY = os.getenv("API_KEY", "default-api-key")  # For authentication

# Global references for SimpleMemory, SimpleLLM, Context7 client
_global_memory_instance: Optional['SimpleMemory'] = None
_global_llm_instance: Optional['SimpleLLM'] = None
_global_context7_mcp_client: Optional[Client] = None

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Simple API key verification"""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except ImportError:
        logging.warning("PyYAML not available, using default configuration")
        return {}
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using default configuration")
        return {}
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

def update_globals_from_config(config: Dict[str, Any]):
    """Update global constants from config"""
    global DEBUG_MODE, OLLAMA_API_URL, MEMORY_JSON_PATH, MAX_MEMORY_ENTRIES
    global LLM_TIMEOUT, CACHE_TTL_SECONDS, OLLAMA_MODEL_NAME, BRAVE_API_KEY
    global CONTEXT7_MCP_SERVER_URL, ARXIV_API_URL, GITHUB_API_URL
    
    if 'server' in config:
        DEBUG_MODE = config['server'].get('debug_mode', DEBUG_MODE)
    
    if 'ollama' in config:
        ollama_config = config['ollama']
        OLLAMA_API_URL = f"http://{ollama_config.get('host', '127.0.0.1')}:{ollama_config.get('port', 11434)}"
        OLLAMA_MODEL_NAME = ollama_config.get('model_name', OLLAMA_MODEL_NAME)
        LLM_TIMEOUT = ollama_config.get('timeout', LLM_TIMEOUT)
    
    if 'memory' in config:
        memory_config = config['memory']
        MEMORY_JSON_PATH = Path(memory_config.get('file_path', str(MEMORY_JSON_PATH))).expanduser()
        MAX_MEMORY_ENTRIES = memory_config.get('max_entries', MAX_MEMORY_ENTRIES)
        CACHE_TTL_SECONDS = memory_config.get('cache_ttl_seconds', CACHE_TTL_SECONDS)
    
    if 'external_services' in config:
        services = config['external_services']
        if 'brave' in services and services['brave'].get('enabled', True):
            BRAVE_API_KEY = services['brave'].get('api_key', BRAVE_API_KEY)
        if 'context7' in services and services['context7'].get('enabled', True):
            CONTEXT7_MCP_SERVER_URL = services['context7'].get('url', CONTEXT7_MCP_SERVER_URL)
        if 'arxiv' in services and services['arxiv'].get('enabled', True):
            ARXIV_API_URL = services['arxiv'].get('api_url', ARXIV_API_URL)
        if 'github' in services and services['github'].get('enabled', True):
            GITHUB_API_URL = services['github'].get('api_url', GITHUB_API_URL)

# --- Classes (SimpleMemory, SimpleLLM) ---
class SimpleMemory:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception as e:
                logging.error(f"Memory load failed: {e}")
        logging.info(f"Memory loaded: {len(self.data)} entries")

    def _save(self):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logging.error(f"Memory save failed: {e}")

    def save(self, key: str, value: str, category: str = "", importance: float = 0.5, tags: List[str] = None):
        if tags is None:
            tags = []
        derived_tags = [t.strip() for t in (category.split(',') if category else [])]
        final_tags = list(set(tags + derived_tags))
        self.data[key] = {
            'value': value,
            'category': category,
            'importance': importance,
            'tags': final_tags,
            'created_at': time.time(),
            'access_count': 0
        }
        self._cleanup()
        self._save()
        return True

    def retrieve(self, key: str) -> Optional[Dict]:
        if entry := self.data.get(key):
            entry['access_count'] += 1
            return entry
        return None

    def search(self, query: str, limit: int = 10, category: str = "") -> List[Dict]:
        results = []
        for key, entry in self.data.items():
            if category and entry['category'] != category:
                continue
            if (query.lower() in key.lower()) or (query.lower() in entry['value'].lower()) or any(query.lower() in tag.lower() for tag in entry.get('tags', [])):
                results.append((key, entry))
        results.sort(key=lambda x: x[1]['importance'], reverse=True)
        return [{'key': k, **v} for k, v in results[:limit]]

    def _cleanup(self):
        if len(self.data) <= MAX_MEMORY_ENTRIES:
            return
        entries = sorted(self.data.items(), key=lambda x: x[1]['access_count'])
        for key, _ in entries[:len(self.data) - MAX_MEMORY_ENTRIES]:
            del self.data[key]
        logging.info(f"Memory cleanup: removed {len(entries) - MAX_MEMORY_ENTRIES} entries")

class SimpleLLM:
    def __init__(self):
        self.cache: Dict[str, str] = {}
        self.cache_timestamps: Dict[str, float] = {}
        # Ollama server init
        start_ollama_server()
        logging.info("Ollama server started successfully.")
        self.model_name = self._discover_model()
        logging.info(f"Using model: {self.model_name or 'None'}")

    def _discover_model(self) -> Optional[str]:
        try:
            res = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
            res.raise_for_status()
            if models := res.json().get('models'):
                if OLLAMA_MODEL_NAME and any(m['name'] == OLLAMA_MODEL_NAME for m in models):
                    return OLLAMA_MODEL_NAME
                return models[0]['name']
        except requests.exceptions.RequestException as e:
            logging.error(f"LLM Model discovery failed: {e}")
        return None

    def call(self, prompt: str, format_json: bool = False) -> str:
        if not self.model_name:
            logging.error("LLM call failed: No available model.")
            return json.dumps({"error": "No available model"})
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self.cache and time.time() - self.cache_timestamps[cache_key] < CACHE_TTL_SECONDS:
            if DEBUG_MODE:
                logging.debug(f"Cache hit: {cache_key[:8]}")
            return self.cache[cache_key]
        try:
            payload = {"model": self.model_name, "prompt": prompt, "stream": False, "system": "", "temperature": 0.7}
            res = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload, timeout=LLM_TIMEOUT)
            res.raise_for_status()
            response = res.json().get("response", "").strip()
            if format_json:
                response = self._extract_json(response)
            self.cache[cache_key] = response
            self.cache_timestamps[cache_key] = time.time()
            return response
        except requests.exceptions.Timeout as e:
            return json.dumps({"error_type": "transient", "message": str(e)})
        except requests.exceptions.RequestException as e:
            return json.dumps({"error_type": "system", "message": str(e)})
        except Exception as e:
            return json.dumps({"error_type": "system", "message": str(e)})

    def _extract_json(self, text: str) -> str:
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        brace_count = 0
        start = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    try:
                        json_str = text[start:i+1]
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        continue
        logging.warning("No valid JSON found in LLM response after multiple attempts.")
        return json.dumps({"error": "Invalid JSON format in LLM response"})

# --- Code Analysis Engine (using existing LLM infrastructure) ---
@dataclass
class CodeEntity:
    type: str  # 'function', 'class', 'variable'
    name: str
    file_path: Path
    line: int
    dependencies: List[str]
    calls: List[str]
    complexity: int = 0

class SimpleCodeAnalyzer:
    """Lightweight code analyzer that integrates with existing LLM/Memory"""
    def __init__(self):
        self.cache = {}
        self.file_timestamps = {}

    def analyze_file(self, file_path: Path) -> Dict[str, CodeEntity]:
        """AST analysis of single Python file"""
        if not file_path.exists():
            return {}
            
        # Check cache validity
        cache_key = str(file_path)
        current_time = os.path.getmtime(file_path)
        
        if (cache_key in self.cache and 
            self.file_timestamps.get(cache_key, 0) == current_time):
            if DEBUG_MODE:
                logging.debug(f"Using cached analysis for {file_path.name}")
            return self.cache[cache_key]
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            analyzer = ASTWalker(file_path)
            analyzer.visit(tree)
            
            # Cache results
            self.cache[cache_key] = analyzer.entities
            self.file_timestamps[cache_key] = current_time
            
            return analyzer.entities
            
        except SyntaxError as e:
            logging.warning(f"Syntax error in {file_path}: {e}")
            return {}
        except Exception as e:
            logging.error(f"Analysis failed for {file_path}: {e}")
            return {}

    def analyze_project(self, project_root: Path, pattern: str = "") -> Dict[str, Any]:
        """Analyze entire project with pattern filtering"""
        all_entities = {}
        py_files = list(project_root.rglob("*.py"))
        
        logging.info(f"Analyzing {len(py_files)} Python files...")
        
        for file_path in py_files:
            entities = self.analyze_file(file_path)
            all_entities.update(entities)
        
        # Filter by pattern if provided
        if pattern:
            filtered = {k: v for k, v in all_entities.items() 
                       if pattern.lower() in k.lower() or 
                          pattern.lower() in v.name.lower()}
            all_entities = filtered
        
        return {
            "entities": all_entities,
            "summary": {
                "total_files": len(py_files),
                "total_entities": len(all_entities),
                "functions": sum(1 for e in all_entities.values() if e.type == "function"),
                "classes": sum(1 for e in all_entities.values() if e.type == "class")
            }
        }

class ASTWalker(ast.NodeVisitor):
    """AST visitor that extracts code entities and relationships"""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.entities = {}
        self.current_class = None
        self.current_function = None
        self.imports = []

    def visit_FunctionDef(self, node):
        entity_key = f"{self.file_path}:{node.name}"
        entity = CodeEntity(
            type="function",
            name=node.name,
            file_path=self.file_path,
            line=node.lineno,
            dependencies=self.imports.copy(),
            calls=[],
            complexity=self._calculate_complexity(node)
        )
        
        # Track function calls within this function
        old_function = self.current_function
        self.current_function = entity
        self.generic_visit(node)
        self.current_function = old_function
        
        self.entities[entity_key] = entity

    def visit_ClassDef(self, node):
        entity_key = f"{self.file_path}:{node.name}"
        entity = CodeEntity(
            type="class",
            name=node.name,
            file_path=self.file_path,
            line=node.lineno,
            dependencies=self.imports.copy(),
            calls=[]
        )
        
        old_class = self.current_class
        self.current_class = entity
        self.generic_visit(node)
        self.current_class = old_class
        
        self.entities[entity_key] = entity

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                self.imports.append(f"{node.module}.{alias.name}")

    def visit_Call(self, node):
        if self.current_function:
            if isinstance(node.func, ast.Name):
                self.current_function.calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                self.current_function.calls.append(node.func.attr)
        self.generic_visit(node)

    def _calculate_complexity(self, node) -> int:
        """Simple cyclomatic complexity calculation"""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, 
                                ast.Try, ast.ExceptHandler)):
                complexity += 1
        return complexity

# Global code analyzer instance
_global_code_analyzer = SimpleCodeAnalyzer()

# --- Helper Functions (unchanged) ---
def json_response(data: Any, status: str = "ok") -> str:
    return json.dumps({"status": status, "data": data}, indent=2)

def init_system(llm_instance: SimpleLLM, memory_instance: SimpleMemory):
    try:
        response = llm_instance.call("System check: respond with 'OK'")
        memory_instance.save("system_init", f"Initialized at {time.ctime()} with model {llm_instance.model_name}", category="system", importance=1.0)
        logging.info(f"System check response: {response[:50]}...")
    except Exception as e:
        logging.error(f"System check failed: {e}")

def safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def is_ollama_running(host='127.0.0.1', port=11434) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect((host, port))
    except (socket.timeout, ConnectionRefusedError):
        return False
    else:
        s.close()
        return True

def start_ollama_server():
    command = ["ollama", "serve"]
    try:
        logging.info("Attempting to start Ollama server in the background...")
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        if is_ollama_running():
            logging.info("Ollama server started successfully.")
            return True
        else:
            logging.error("Ollama server failed to start after command execution.")
            return False
    except FileNotFoundError:
        logging.error("The 'ollama' command was not found. Please ensure Ollama is installed and in your system's PATH.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to start the Ollama server: {e}")
        return False

def search_brave(query: str, max_results: int) -> List[Dict]:
    if not BRAVE_API_KEY:
        return []
    try:
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
        params = {"q": query, "count": max_results}
        response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return [{"title": r.get("title"), "url": r.get("url")} for r in response.json().get("web", {}).get("results", [])]
    except requests.exceptions.RequestException as e:
        logging.error(f"Brave search error: {e}")
    return []

def search_arxiv(query: str, max_results: int) -> List[Dict]:
    try:
        params = {"search_query": query, "max_results": max_results}
        response = requests.get(ARXIV_API_URL, params=params, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        return [{"title": entry.title, "url": entry.link} for entry in feed.entries]
    except requests.exceptions.RequestException as e:
        logging.error(f"arXiv search error: {e}")
    return []

def search_github(query: str, max_results: int) -> List[Dict]:
    try:
        params = {"q": query, "per_page": max_results}
        response = requests.get(GITHUB_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return [{"title": item["name"], "url": item["html_url"]} for item in response.json().get("items", [])]
    except requests.exceptions.RequestException as e:
        logging.error(f"GitHub search error: {e}")
    return []

# Create FastAPI app
app = FastAPI(
    title="AgenticCore MCP Server",
    description="AI-powered research and code analysis server with FastAPI and FastMCP integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI Routes
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with server information"""
    return {
        "name": "AgenticCore MCP Server",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": time.time(),
        "endpoints": {
            "health": "/health",
            "search": "/api/search",
            "memory": "/api/memory",
            "analyze": "/api/analyze-codebase",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check_endpoint(detailed: bool = False):
    """Comprehensive health check endpoint"""
    result = await health_check_func(detailed=detailed)
    health_data = json.loads(result)["data"]
    
    if health_data["status"] == "unhealthy":
        return JSONResponse(status_code=503, content=health_data)
    return health_data

@app.post("/api/search", tags=["AI Tools"])
async def search_endpoint(request: SearchRequest, token: str = Depends(verify_token)):
    """Unified search across multiple sources with synthesis"""
    try:
        result = await unified_search_tool_func(
            query=request.query,
            sources=request.sources,
            synthesize=request.synthesize,
            max_results=request.max_results
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logging.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory", tags=["AI Tools"])
async def memory_endpoint(request: MemoryRequest, token: str = Depends(verify_token)):
    """Memory management endpoint"""
    try:
        result = memory_tool_func(
            action=request.action,
            key=request.key or "",
            value=request.value or "",
            query=request.query or "",
            category=request.category,
            tags=request.tags
        )
        return JSONResponse(content={"success": True, "result": json.loads(result)})
    except Exception as e:
        logging.error(f"Memory endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-codebase", tags=["Code Analysis"])
async def analyze_codebase_endpoint(request: CodebaseAnalysisRequest, token: str = Depends(verify_token)):
    """Codebase analysis endpoint"""
    try:
        result = analyze_codebase_func(
            project_root=request.project_root,
            pattern=request.pattern,
            max_files=request.max_files
        )
        return JSONResponse(content={"success": True, "result": json.loads(result)})
    except Exception as e:
        logging.error(f"Codebase analysis endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status", tags=["System"])
async def system_status_endpoint(include_stats: bool = True, token: str = Depends(verify_token)):
    """Get detailed system status"""
    try:
        result = system_status_func(include_stats=include_stats)
        return JSONResponse(content=json.loads(result))
    except Exception as e:
        logging.error(f"System status endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs", tags=["System"])
async def logs_endpoint(lines: int = 50, token: str = Depends(verify_token)):
    """Read recent log entries"""
    try:
        result = read_logs_func(lines=lines)
        return JSONResponse(content={"logs": result})
    except Exception as e:
        logging.error(f"Logs endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- [FIXED] Tool Functions using Direct Global Access ---
async def unified_search_tool_func(query: str, sources: List[str] = [], synthesize: bool = True, max_results: int = 5, ctx: Context = None) -> str:
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    if not sources:  # Changed from sources is None
        sources = ["brave", "context7", "arxiv", "github", "memory"]
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        sync_futures = {s: executor.submit(globals()[f"search_{s}"], query, max_results) for s in sources if s in ["brave", "arxiv", "github"]}
        if "memory" in sources:
            sync_futures["memory"] = executor.submit(_global_memory_instance.search, query, max_results)
        for source, future in sync_futures.items():
            results[source] = future.result()

    # Direct Context7 logic integration
    if "context7" in sources and _global_context7_mcp_client:
        logging.info(f"Calling official Context7 server for query: {query}")
        try:
            search_result = await _global_context7_mcp_client.call_tool("search", {"query": query, "limit": max_results})
            if not search_result.is_error:
                results["context7"] = search_result.data.get("results", []) # Assuming the tool returns a 'results' key
            else:
                results["context7"] = []
        except Exception as e:
            logging.error(f"Context7 search within unified_search failed: {e}")
            results["context7"] = []

    if synthesize:
        prompt = f"Synthesize info about: {query}\nSources:\n{json.dumps(results, indent=2)}\n\nProvide a comprehensive response."
        synthesized = _global_llm_instance.call(prompt)
        _global_memory_instance.save(f"search_{hashlib.md5(query.encode()).hexdigest()}", synthesized, "search", 0.8)
        return synthesized

    return json_response(results)

async def knowledge_graph_builder_func(query: str, sources: List[str] = [], depth: int = 2, ctx: Context = None) -> str:
    if not sources:  # Changed from sources is None
        sources = ["arxiv", "github", "memory"]
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    search_results_str = await unified_search_tool_func(query, sources, synthesize=False, ctx=ctx)
    search_results = json.loads(search_results_str)
    prompt = f'From search results about {query}, extract entities and relationships (depth: {depth}). Format as JSON with "entities" and "relationships" keys.\n\nResults:\n{json.dumps(search_results, indent=2)}'
    graph_data = _global_llm_instance.call(prompt, format_json=True)
    _global_memory_instance.save(f"kg_{hashlib.md5(query.encode()).hexdigest()}", graph_data, "knowledge_graph", 0.9)
    return graph_data

def document_analyzer_func(file_path: str, chunk_size: int = 1000, analyze: bool = True, ctx: Context = None) -> str:
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    try:
        resolved_path = Path(file_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if resolved_path.suffix.lower() == ".pdf":
            with open(resolved_path, "rb") as f:
                text = "\n".join([p.extract_text() for p in pypdf.PdfReader(f).pages if p.extract_text()])
        elif resolved_path.suffix.lower() in [".txt", ".md", ".json", ".csv"]:
            with open(resolved_path, "r", encoding='utf-8') as f:
                text = f.read()
        else:
            return json_response({"error": f"Unsupported file type: {resolved_path.suffix}"}, "error")
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        if analyze:
            prompt = f"Analyze document chunks for themes, entities, and actions.\n\nChunks:\n{json.dumps(chunks, indent=2)}\n\nProvide a concise analysis."
            analysis = _global_llm_instance.call(prompt)
            _global_memory_instance.save(f"doc_{resolved_path.name}", analysis, "document", 0.85)
            return analysis
        return json_response({"chunks": chunks, "total_chunks": len(chunks)})
    except Exception as e:
        return json_response({"error": str(e)}, "error")

def academic_research_tool_func(query: str, sources: List[str] = ["arxiv"], citation_analysis: bool = True, ctx: Context = None) -> str:
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    results = {}
    if "arxiv" in sources:
        results["arxiv"] = search_arxiv(query, 5)
    if citation_analysis:
        prompt = f"Perform citation analysis on papers about {query}.\n\nPapers:\n{json.dumps(results, indent=2)}\n\nIdentify key papers, authors, trends, and gaps."
        analysis = _global_llm_instance.call(prompt)
        _global_memory_instance.save(f"academic_{hashlib.md5(query.encode()).hexdigest()}", analysis, "academic", 0.9)
        return analysis
    return json_response(results)

async def fetch_documentation_func(library_name: str, topic: str, ctx: Context = None) -> str:
    """
    Resolves a library name to an ID and fetches its documentation on a specific topic.
    This is a composite tool that chains 'resolve-library-id' and 'get-library-docs'.
    """
    if _global_context7_mcp_client is None:
        return json_response({"error": "Context7 client not initialized."}, "error")
    try:
        # Step 1: Call the 'resolve-library-id' tool on the Context7 server
        logging.info(f"Resolving library ID for '{library_name}'...")
        resolve_result = await _global_context7_mcp_client.call_tool(
            "resolve-library-id",
            {"libraryName": library_name}
        )
        if resolve_result.is_error or not isinstance(resolve_result.data, dict):
            logging.error(f"Failed to resolve library ID: {resolve_result.content}")
            return json_response({"error": "Could not resolve library ID.", "details": resolve_result.content}, "error")
        library_id = resolve_result.data.get("libraryId")
        if not library_id:
            return json_response({"error": "No valid libraryId found in resolution response."}, "error")

        logging.info(f"Resolved '{library_name}' to ID: {library_id}. Now fetching docs for topic: '{topic}'...")
        # Step 2: Call the 'get-library-docs' tool with the resolved ID
        docs_result = await _global_context7_mcp_client.call_tool(
            "get-library-docs",
            {"context7CompatibleLibraryID": library_id, "topic": topic}
        )
        if docs_result.is_error:
            logging.error(f"Failed to get library docs: {docs_result.content}")
            return json_response({"error": "Could not retrieve documentation.", "details": docs_result.content}, "error")
        # Return the final documentation content
        return json.dumps(docs_result.data) if isinstance(docs_result.data, dict) else str(docs_result.data)
    except Exception as e:
        logging.critical(f"An unexpected error occurred in fetch_documentation_func: {e}", exc_info=True)
        return json_response({"error": f"An unexpected error occurred: {e}"}, "error")

def local_web_scraper_func(url: str, depth: int = 1, structured: bool = True, ctx: Context = None) -> str:
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for item in soup(["script", "style"]):
            item.decompose()
        text = soup.get_text(separator=' ', strip=True)
        if structured:
            prompt = f'Extract title, main_content, key_points, and metadata from this webpage content.\n\nContent:\n{text[:8000]}...'
            structured_data = _global_llm_instance.call(prompt, format_json=True)
            _global_memory_instance.save(f"web_{hashlib.md5(url.encode()).hexdigest()}", structured_data, "web", 0.7)
            return structured_data
        return json_response({"content": text})
    except Exception as e:
        return json_response({"error": str(e)}, "error")

def memory_tool_func(action: str, key: str = "", value: str = "", query: str = "", category: str = "", tags: List[str] = [], ctx: Context = None) -> str:
    if _global_memory_instance is None:
        return json_response({"error": "Memory not initialized"}, "error")
    if action == "save":
        if not key or not value:
            return json_response({"error": "Key and value required for save."}, "error")
        derived_tags = [t.strip() for t in (category.split(',') if category else [])]
        final_tags = list(set(tags + derived_tags))
        return json_response({"success": _global_memory_instance.save(key, value, category, tags=final_tags), "key": key})
    elif action == "retrieve":
        if not key:
            return json_response({"error": "Key required for retrieve."}, "error")
        return json_response(_global_memory_instance.retrieve(key) or {"message": f"No entry for key: {key}"})
    elif action == "search":
        if not query:
            return json_response({"error": "Query required for search."}, "error")
        return json_response(_global_memory_instance.search(query, limit=10, category=category))
    else:
        return json_response({"error": "Invalid action. Choose 'save', 'retrieve', or 'search'."}, "error")

def chat_tool_func(message: str, context: str = "", style: str = "balanced", ctx: Context = None) -> str:
    if _global_llm_instance is None:
        return json_response({"error": "LLM not initialized"}, "error")
    prompt = f"Message: {message}\n"
    if context:
        prompt += f"Context: {context}\n"
    prompt += f"Response style: {style}. Be concise and relevant."
    return _global_llm_instance.call(prompt)

def read_logs_func(lines: int = 50, ctx: Context = None) -> str:
    log_file = Path("logs") / "agentic_core_server.log"
    if not log_file.exists():
        return json_response({"error": "Log file not found"}, "error")
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            return "".join(f.readlines()[-lines:])
    except Exception as e:
        return json_response({"error": f"Failed to read log file: {e}"}, "error")

async def cognitive_gridlock_protocol_func(central_dilemma: str, competing_assumptions: List[str], amplification_cycles: int = 3, ctx: Context = None) -> dict:
    if _global_llm_instance is None:
        return json_response({"error": "LLM not initialized"}, "error")
    prompt = (f"Analyze gridlock scenario:\nCentral Dilemma: {central_dilemma}\n"
              f"Competing Assumptions:\n" + "\n".join(f"- {a}" for a in competing_assumptions) + "\n"
              f"Perform {amplification_cycles} cycles of recursive analysis to identify irreducible trade-offs.")
    analysis = _global_llm_instance.call(prompt)
    return {
        "report": f"## Strategic Impasse Report\n\n**Central Dilemma**: {central_dilemma}\n\n**Competing Assumptions**:\n- " + "\n- ".join(competing_assumptions) + f"\n\n**Irreducible Trade-offs**:\n{analysis}\n\n**Resolution Strategy**: Hybrid approach",
        "status": "analysis_complete"
    }

async def health_check_func(detailed: bool = False, ctx: Context = None) -> str:
    """Comprehensive health check for all server components"""
    start_time = time.time()
    checks = {}
    
    # Server context check
    try:
        if _global_memory_instance is None or _global_llm_instance is None:
            checks["server_context"] = {"status": "error", "error": "Global instances not initialized"}
        else:
            checks["server_context"] = {"status": "healthy"}
    except Exception as e:
        checks["server_context"] = {"status": "error", "error": str(e)}
    
    # Ollama check
    try:
        ollama_url = f"{OLLAMA_API_URL}/api/tags"
        response = requests.get(ollama_url, timeout=5)
        if response.status_code == 200:
            checks["ollama"] = {
                "status": "healthy",
                "host": OLLAMA_API_URL.split("//")[1].split(":")[0],
                "port": int(OLLAMA_API_URL.split(":")[-1]),
                "url": OLLAMA_API_URL
            }
        else:
            checks["ollama"] = {"status": "error", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        checks["ollama"] = {"status": "error", "error": str(e)}
    
    # Memory check
    try:
        if _global_memory_instance:
            checks["memory"] = {
                "status": "healthy",
                "entries": len(_global_memory_instance.data),
                "file_path": str(_global_memory_instance.file_path)
            }
        else:
            checks["memory"] = {"status": "error", "error": "Memory not initialized"}
    except Exception as e:
        checks["memory"] = {"status": "error", "error": str(e)}
    
    # External services checks
    for service_name in ["brave", "context7", "arxiv", "github"]:
        try:
            if service_name == "brave":
                checks[f"service_{service_name}"] = {"status": "not_tested"}
            elif service_name == "context7":
                if _global_context7_mcp_client:
                    checks[f"service_{service_name}"] = {"status": "healthy"}
                else:
                    checks[f"service_{service_name}"] = {"status": "error", "error": "Client not initialized"}
            else:
                checks[f"service_{service_name}"] = {"status": "not_tested"}
        except Exception as e:
            checks[f"service_{service_name}"] = {"status": "error", "error": str(e)}
    
    # AARU engine check
    try:
        # Check if aaru_engine.py exists and can be imported
        import aaru_engine
        checks["aaru_engine"] = {"status": "healthy"}
    except ImportError:
        checks["aaru_engine"] = {"status": "error", "error": "AARU engine not available"}
    except Exception as e:
        checks["aaru_engine"] = {"status": "error", "error": str(e)}
    
    # Calculate summary
    total_checks = len(checks)
    healthy_checks = sum(1 for check in checks.values() if check.get("status") == "healthy")
    health_percentage = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
    
    # Collect issues
    issues = []
    for check_name, check_data in checks.items():
        if check_data.get("status") == "error":
            issues.append(f"{check_name} error: {check_data.get('error', 'Unknown error')}")
    
    result = {
        "status": "healthy" if health_percentage >= 80 else "unhealthy",
        "timestamp": time.time(),
        "checks": checks,
        "summary": {
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "health_percentage": health_percentage,
            "total_issues": len(issues)
        },
        "issues": issues
    }
    
    return json_response(result)

def system_status_func(include_stats: bool = True, ctx: Context = None) -> str:
    """Get detailed system status and performance metrics"""
    try:
        status_data = {
            "server_info": {
                "name": "AgenticCore MCP Server",
                "version": "2.0.0",
                "uptime": time.time() - (getattr(system_status_func, '_start_time', time.time())),
                "config_loaded": True
            },
            "components": {}
        }
        
        # Memory status
        if _global_memory_instance:
            status_data["components"]["memory"] = {
                "status": "active",
                "entries": len(_global_memory_instance.data),
                "file_path": str(_global_memory_instance.file_path),
                "max_entries": MAX_MEMORY_ENTRIES
            }
        else:
            status_data["components"]["memory"] = {"status": "inactive"}
        
        # LLM status
        if _global_llm_instance:
            status_data["components"]["llm"] = {
                "status": "active",
                "model": _global_llm_instance.model_name,
                "cache_size": len(_global_llm_instance.cache),
                "api_url": OLLAMA_API_URL
            }
        else:
            status_data["components"]["llm"] = {"status": "inactive"}
        
        # Context7 status
        if _global_context7_mcp_client:
            status_data["components"]["context7"] = {
                "status": "active",
                "url": CONTEXT7_MCP_SERVER_URL
            }
        else:
            status_data["components"]["context7"] = {"status": "inactive"}
        
        # Configuration
        status_data["configuration"] = {
            "debug_mode": DEBUG_MODE,
            "ollama_url": OLLAMA_API_URL,
            "memory_path": str(MEMORY_JSON_PATH),
            "max_memory_entries": MAX_MEMORY_ENTRIES,
            "llm_timeout": LLM_TIMEOUT,
            "cache_ttl": CACHE_TTL_SECONDS
        }
        
        if include_stats:
            # Performance stats
            status_data["performance"] = {
                "memory_usage": len(_global_memory_instance.data) if _global_memory_instance else 0,
                "cache_hits": getattr(_global_llm_instance, '_cache_hits', 0) if _global_llm_instance else 0,
                "total_requests": getattr(system_status_func, '_total_requests', 0)
            }
        
        return json_response(status_data)
        
    except Exception as e:
        return json_response({"error": f"System status error: {e}"}, "error")

async def synthesize_insights_func(synthesis_type: str, topic: str, source_identifiers: List[str] = [], output_format: str = "markdown", ctx: Context = None) -> str:
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    prompt = (f"Perform {synthesis_type} synthesis on topic: {topic}\n"
              f"Sources: {', '.join(source_identifiers)}\n"
              f"Generate key insights and actionable recommendations.")
    insights = _global_llm_instance.call(prompt)
    if output_format == "markdown":
        formatted_output = (f"# {topic.capitalize()} Synthesis Report\n\n"
                            f"**Synthesis Type**: {synthesis_type.replace('_', ' ').title()}\n\n"
                            f"**Key Insights**:\n{insights}\n\n"
                            f"**Action Plan**:\n1. Implement phased approach\n2. Validate assumptions\n3. Measure outcomes")
    elif output_format == "json":
        try:
            formatted_output = json.dumps({"topic": topic, "type": synthesis_type, "insights": json.loads(insights), "sources": source_identifiers}, indent=2)
        except json.JSONDecodeError:
            formatted_output = json.dumps({"topic": topic, "type": synthesis_type, "insights": insights, "sources": source_identifiers, "error": "LLM did not return valid JSON."}, indent=2)
    elif output_format == "bullet_points":
        formatted_output = "\n".join([f"- {line.strip()}" for line in insights.splitlines() if line.strip()])
    else:
        formatted_output = insights
    _global_memory_instance.save(f"synthesis_{hashlib.md5(topic.encode()).hexdigest()}", formatted_output, synthesis_type, 0.95)
    return formatted_output

# --- Code Analysis Tool Functions (using existing LLM/Memory infrastructure) ---
def analyze_codebase_func(project_root: str, pattern: str = "", max_files: int = 50, ctx: Context = None) -> str:
    """Analyze codebase using AST + LLM insights"""
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    
    try:
        project_path = Path(project_root).expanduser().resolve()
        if not project_path.exists():
            return json_response({"error": f"Project path not found: {project_root}"}, "error")
        
        logging.info(f"Analyzing codebase: {project_path}")
        
        # AST Analysis
        analysis_result = _global_code_analyzer.analyze_project(project_path, pattern)
        entities = analysis_result["entities"]
        summary = analysis_result["summary"]
        
        # Limit results for LLM processing
        limited_entities = dict(list(entities.items())[:max_files])
        
        # Prepare data for LLM analysis
        entity_data = []
        for key, entity in limited_entities.items():
            entity_data.append({
                "name": entity.name,
                "type": entity.type,
                "file": str(entity.file_path.name),
                "line": entity.line,
                "complexity": entity.complexity,
                "calls": entity.calls[:5],  # Limit calls
                "dependencies": entity.dependencies[:5]  # Limit deps
            })
        
        # LLM Analysis
        if entity_data:
            prompt = f"""
            Analyze this codebase structure for pattern: "{pattern}"
            
            Summary: {json.dumps(summary, indent=2)}
            
            Key Entities: {json.dumps(entity_data[:10], indent=2)}
            
            Provide insights on:
            1. Code organization patterns
            2. Potential refactoring opportunities
            3. High-complexity functions (>10)
            4. Integration points for new features
            5. Architectural recommendations
            
            Focus on actionable insights for developers.
            """
            
            llm_insights = _global_llm_instance.call(prompt)
            
            # Cache in memory
            cache_key = f"codebase_analysis_{hashlib.md5(project_root.encode()).hexdigest()}"
            _global_memory_instance.save(
                cache_key, 
                llm_insights, 
                category="code_analysis",
                tags=["codebase", "ast", "analysis"]
            )
        else:
            llm_insights = "No entities found matching the pattern."
        
        result = {
            "summary": summary,
            "pattern_matches": len(limited_entities),
            "insights": llm_insights,
            "entities": [{"name": e.name, "type": e.type, "file": e.file_path.name, 
                         "line": e.line, "complexity": e.complexity} 
                        for e in list(limited_entities.values())[:20]]
        }
        
        return json_response(result)
        
    except Exception as e:
        logging.error(f"Codebase analysis failed: {e}")
        return json_response({"error": str(e)}, "error")

def trace_function_calls_func(function_name: str, project_root: str, depth: int = 3, ctx: Context = None) -> str:
    """Trace function calls with LLM-powered insights"""
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    
    try:
        project_path = Path(project_root).expanduser().resolve()
        if not project_path.exists():
            return json_response({"error": f"Project path not found: {project_root}"}, "error")
        
        # Analyze project to find function
        analysis_result = _global_code_analyzer.analyze_project(project_path, function_name)
        entities = analysis_result["entities"]
        
        # Find target function
        target_functions = [e for e in entities.values() 
                          if e.name == function_name and e.type == "function"]
        
        if not target_functions:
            return json_response({"error": f"Function '{function_name}' not found"}, "error")
        
        # Build call graph
        call_graph = {}
        for func in target_functions:
            call_graph[f"{func.file_path.name}:{func.name}"] = {
                "line": func.line,
                "complexity": func.complexity,
                "calls": func.calls,
                "dependencies": func.dependencies
            }
        
        # LLM Analysis of call patterns
        prompt = f"""
        Analyze function call patterns for: {function_name}
        
        Call Graph: {json.dumps(call_graph, indent=2)}
        
        Provide analysis on:
        1. Function responsibility and purpose
        2. Coupling with other functions
        3. Potential side effects
        4. Refactoring recommendations
        5. Testing strategies
        
        Focus on code quality and maintainability.
        """
        
        llm_analysis = _global_llm_instance.call(prompt)
        
        # Cache results
        cache_key = f"call_trace_{function_name}_{hashlib.md5(project_root.encode()).hexdigest()}"
        _global_memory_instance.save(
            cache_key,
            llm_analysis,
            category="function_analysis", 
            tags=["calls", "trace", function_name]
        )
        
        result = {
            "function_name": function_name,
            "found_instances": len(target_functions),
            "call_graph": call_graph,
            "analysis": llm_analysis
        }
        
        return json_response(result)
        
    except Exception as e:
        logging.error(f"Function trace failed: {e}")
        return json_response({"error": str(e)}, "error")

def refactor_safely_func(function_name: str, project_root: str, proposed_changes: str, ctx: Context = None) -> str:
    """Safe refactoring analysis with impact assessment"""
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    
    try:
        project_path = Path(project_root).expanduser().resolve()
        
        # Analyze current state
        analysis_result = _global_code_analyzer.analyze_project(project_path, function_name)
        entities = analysis_result["entities"]
        
        # Find function and its callers
        target_function = None
        callers = []
        
        for entity in entities.values():
            if entity.name == function_name and entity.type == "function":
                target_function = entity
            elif function_name in entity.calls:
                callers.append({
                    "name": entity.name,
                    "file": str(entity.file_path.name),
                    "line": entity.line,
                    "type": entity.type
                })
        
        if not target_function:
            return json_response({"error": f"Function '{function_name}' not found"}, "error")
        
        # LLM Impact Analysis
        prompt = f"""
        Analyze refactoring safety for function: {function_name}
        
        Current Function:
        - File: {target_function.file_path.name}
        - Line: {target_function.line}
        - Complexity: {target_function.complexity}
        - Dependencies: {target_function.dependencies[:10]}
        
        Callers ({len(callers)}): {json.dumps(callers[:10], indent=2)}
        
        Proposed Changes: {proposed_changes}
        
        Assess:
        1. Breaking change risk (1-10 scale)
        2. Files that need updates
        3. Testing requirements
        4. Backwards compatibility options
        5. Step-by-step refactoring plan
        
        Provide specific, actionable recommendations.
        Format as JSON with risk_score, affected_files, recommendations, and plan.
        """
        
        safety_analysis = _global_llm_instance.call(prompt, format_json=True)
        
        # Store refactoring analysis
        cache_key = f"refactor_{function_name}_{int(time.time())}"
        _global_memory_instance.save(
            cache_key,
            safety_analysis,
            category="refactoring",
            tags=["safety", "impact", function_name]
        )
        
        result = {
            "function_name": function_name,
            "current_location": f"{target_function.file_path.name}:{target_function.line}",
            "caller_count": len(callers),
            "complexity": target_function.complexity,
            "safety_analysis": safety_analysis,
            "callers": callers[:10]  # Limit output
        }
        
        return json_response(result)
        
    except Exception as e:
        logging.error(f"Refactoring analysis failed: {e}")
        return json_response({"error": str(e)}, "error")

def validate_codebase_func(project_root: str, check_complexity: bool = True, complexity_threshold: int = 10, ctx: Context = None) -> str:
    """Validate codebase for common issues and quality metrics"""
    if _global_llm_instance is None or _global_memory_instance is None:
        return json_response({"error": "System not initialized"}, "error")
    
    try:
        project_path = Path(project_root).expanduser().resolve()
        
        # Full codebase analysis
        analysis_result = _global_code_analyzer.analyze_project(project_path)
        entities = analysis_result["entities"]
        summary = analysis_result["summary"]
        
        issues = []
        
        # Check complexity
        if check_complexity:
            complex_functions = [e for e in entities.values() 
                               if e.type == "function" and e.complexity > complexity_threshold]
            for func in complex_functions:
                issues.append({
                    "type": "high_complexity",
                    "severity": "medium",
                    "function": func.name,
                    "file": str(func.file_path.name),
                    "line": func.line,
                    "complexity": func.complexity,
                    "message": f"Function has complexity {func.complexity} (threshold: {complexity_threshold})"
                })
        
        # Check for functions with many dependencies
        for entity in entities.values():
            if entity.type == "function" and len(entity.dependencies) > 15:
                issues.append({
                    "type": "high_coupling",
                    "severity": "medium", 
                    "function": entity.name,
                    "file": str(entity.file_path.name),
                    "line": entity.line,
                    "dependencies": len(entity.dependencies),
                    "message": f"Function has {len(entity.dependencies)} dependencies"
                })
        
        # LLM Quality Analysis
        if issues:
            prompt = f"""
            Analyze code quality issues:
            
            Summary: {json.dumps(summary, indent=2)}
            
            Issues Found: {json.dumps(issues[:10], indent=2)}
            
            Provide recommendations for:
            1. Priority order for fixes
            2. Refactoring strategies
            3. Code quality improvements
            4. Architectural suggestions
            
            Focus on maintainability and technical debt reduction.
            """
            
            quality_recommendations = _global_llm_instance.call(prompt)
        else:
            quality_recommendations = "No significant quality issues found. Code appears well-structured."
        
        result = {
            "summary": summary,
            "total_issues": len(issues),
            "issues_by_type": {
                "high_complexity": len([i for i in issues if i["type"] == "high_complexity"]),
                "high_coupling": len([i for i in issues if i["type"] == "high_coupling"])
            },
            "issues": issues[:20],  # Limit output
            "recommendations": quality_recommendations
        }
        
        return json_response(result)
        
    except Exception as e:
        logging.error(f"Codebase validation failed: {e}")
        return json_response({"error": str(e)}, "error")

# --- DEFERRED INITIALIZATION: performed in main_async() to avoid side effects at import time ---
_global_memory_instance = None
_global_llm_instance = None

# Create the FastMCP server instance from FastAPI app
mcp_server = FastMCP.from_fastapi(app)

# --- TOOL REGISTRATION (no changes needed) ---
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(unified_search_tool_func), name="unified_search", description="Performs a parallel search across multiple sources and synthesizes the results.", transform_args={'query': ArgTransform(description="The search topic or question."), 'sources': ArgTransform(description="Optional list of sources: 'brave', 'context7', 'arxiv', 'github', 'memory'."), 'synthesize': ArgTransform(description="If True, synthesizes results into a summary. If False, returns raw JSON."), 'max_results': ArgTransform(description="Maximum number of results to fetch from each source.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(knowledge_graph_builder_func), name="knowledge_graph_builder", description="Builds a knowledge graph from search results to show entities and relationships.", transform_args={'query': ArgTransform(description="The central topic for the knowledge graph."), 'sources': ArgTransform(description="Sources for info. Defaults to 'arxiv', 'github', 'memory'."), 'depth': ArgTransform(description="Desired level of detail for entity extraction.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(document_analyzer_func), name="document_analyzer", description="Reads and analyzes the content of a local document (PDF, TXT, MD).", transform_args={'file_path': ArgTransform(description="The local path to the document file."), 'chunk_size': ArgTransform(description="The size of text chunks to split the document into."), 'analyze': ArgTransform(description="If True, performs an LLM-based analysis. If False, returns raw text chunks.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(academic_research_tool_func), name="academic_research", description="Conducts research focused on academic sources like arXiv and performs citation analysis.", transform_args={'query': ArgTransform(description="The research topic."), 'sources': ArgTransform(description="A list of academic sources to search. Currently supports 'arxiv'."), 'citation_analysis': ArgTransform(description="If True, performs LLM analysis of citations, trends, and gaps.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(local_web_scraper_func), name="web_scraper", description="Scrapes textual content from a URL, returning structured data or raw text.", transform_args={'url': ArgTransform(description="The URL of the web page to scrape."), 'depth': ArgTransform(description="Placeholder for future recursive scraping (currently unused)."), 'structured': ArgTransform(description="If True, extracts structured JSON data. If False, returns raw text.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(memory_tool_func), name="memory", description="Manages the agent's long-term memory (save, retrieve, search).", transform_args={'action': ArgTransform(description="The action to perform: 'save', 'retrieve', or 'search'."), 'key': ArgTransform(description="The unique ID for a memory (for 'save' and 'retrieve')."), 'value': ArgTransform(description="The information to be stored (for 'save')."), 'query': ArgTransform(description="The search term to find relevant memories (for 'search')."), 'category': ArgTransform(description="An optional category to group memories."), 'tags': ArgTransform(description="Optional tags for more granular searching.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(chat_tool_func), name="chat", description="Engages in a direct conversation with the LLM.", transform_args={'message': ArgTransform(description="The user's message or question."), 'context': ArgTransform(description="Optional context for a more informed response."), 'style': ArgTransform(description="Desired response style (e.g., 'concise', 'detailed').")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(read_logs_func), name="read_logs", description="Reads the last N lines from the agent's log file for debugging.", transform_args={'lines': ArgTransform(description="The number of recent log lines to retrieve.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(cognitive_gridlock_protocol_func), name="cognitive_gridlock_protocol", description="Analyzes a strategic dilemma to identify irreducible trade-offs.", transform_args={'central_dilemma': ArgTransform(description="A clear statement of the core problem or conflict."), 'competing_assumptions': ArgTransform(description="A list of conflicting assumptions or viewpoints."), 'amplification_cycles': ArgTransform(description="The number of recursive analysis cycles to perform.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(synthesize_insights_func), name="synthesize_insights", description="Synthesizes information from multiple sources into a higher-level summary or plan.", transform_args={'synthesis_type': ArgTransform(description="The type of synthesis to perform (e.g., 'action_plan', 'executive_summary')."), 'source_identifiers': ArgTransform(description="List of sources to synthesize (e.g., 'memory:my_key', 'file:path/to/doc.md')."), 'topic': ArgTransform(description="The central subject or question to focus the synthesis on."), 'output_format': ArgTransform(description="Desired format for the output: 'markdown', 'json', or 'bullet_points'.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(analyze_codebase_func), name="analyze_codebase", description="Analyzes the structure and complexity of a codebase using AST and LLM insights.", transform_args={'project_root': ArgTransform(description="The root directory of the codebase to analyze."), 'pattern': ArgTransform(description="Optional: a pattern to filter files by name."), 'max_files': ArgTransform(description="Optional: the maximum number of files to analyze.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(trace_function_calls_func), name="trace_function_calls", description="Traces function calls within a codebase to identify coupling and potential issues.", transform_args={'function_name': ArgTransform(description="The name of the function to trace."), 'project_root': ArgTransform(description="The root directory of the codebase."), 'depth': ArgTransform(description="Optional: the depth of function calls to trace.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(refactor_safely_func), name="refactor_safely", description="Analyzes the safety of proposed code refactoring changes.", transform_args={'function_name': ArgTransform(description="The name of the function to refactor."), 'project_root': ArgTransform(description="The root directory of the codebase."), 'proposed_changes': ArgTransform(description="A description of the proposed changes.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(validate_codebase_func), name="validate_codebase", description="Validates a codebase for common issues, complexity, and quality metrics.", transform_args={'project_root': ArgTransform(description="The root directory of the codebase to validate."), 'check_complexity': ArgTransform(description="If True, checks for high-complexity functions."), 'complexity_threshold': ArgTransform(description="The threshold for function complexity.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(fetch_documentation_func), name="fetch_documentation", description="Fetches documentation for a software library on a specific topic by auto-resolving its ID.", transform_args={'library_name': ArgTransform(description="The common name of the library (e.g., 'react', 'pandas')."), 'topic': ArgTransform(description="The specific topic to search for in the documentation (e.g., 'hooks', 'dataframe').")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(health_check_func), name="health_check", description="Performs comprehensive health check of all server components and dependencies.", transform_args={'detailed': ArgTransform(description="Whether to perform detailed checks including external service connectivity tests.")}))
mcp_server.add_tool(Tool.from_tool(FunctionTool.from_function(system_status_func), name="system_status", description="Get detailed system status, performance metrics, and resource usage information.", transform_args={'include_stats': ArgTransform(description="Whether to include detailed performance statistics and metrics.")}))

# --- MAIN ASYNC FUNCTION for agentic_core_server.py ---
async def main_async():
    global _global_context7_mcp_client # Declare this global here for modification
    parser = argparse.ArgumentParser(description="AgenticCore MCP Server with FastAPI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--ollama-host", default="127.0.0.1", help="Ollama server host")
    parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--ollama-model", type=str, help="Ollama model name (e.g., 'llama2')", default="Gemma-2:latest")
    parser.add_argument("--context7-url", type=str, default="https://mcp.context7.com/mcp", help="Context7 MCP server URL")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Load configuration from file
    config = load_config(args.config)
    update_globals_from_config(config)
    
    # Update module-level globals based on parsed arguments (command line overrides config)
    global DEBUG_MODE, OLLAMA_API_URL, OLLAMA_MODEL_NAME
    DEBUG_MODE = args.debug or DEBUG_MODE
    OLLAMA_API_URL = f"http://{args.ollama_host}:{args.ollama_port}"
    OLLAMA_MODEL_NAME = args.ollama_model if args.ollama_model else OLLAMA_MODEL_NAME
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting AgenticCore MCP Server with FastAPI")
    
    # Re-initialize global LLM and Memory instances with potentially new OLLAMA_API_URL/MODEL_NAME
    global _global_llm_instance, _global_memory_instance
    try:
        _global_memory_instance = SimpleMemory(MEMORY_JSON_PATH)
        _global_llm_instance = SimpleLLM()

        if not is_ollama_running(host=args.ollama_host, port=args.ollama_port):
            logging.warning("Ollama server not detected.")
            if not start_ollama_server():
                logging.critical("Could not start the Ollama server. Please start it manually and restart this script.")
                sys.exit(1)
        else:
            logging.info("Ollama server is already running.")
        init_system(_global_llm_instance, _global_memory_instance)
        if args.context7_url:
            try:
                _global_context7_mcp_client = Client(args.context7_url)
                logging.info(f"Context7 MCP client initialized for: {args.context7_url}")
            except Exception as e:
                logging.error(f"Failed to initialize Context7 MCP client: {e}")
                _global_context7_mcp_client = None

    except Exception as e:
        logging.critical(f"Failed to initialize server core components: {e}", exc_info=True)
        sys.exit(1)
    
    # Start the server using uvicorn
    uvicorn_config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="debug" if DEBUG_MODE else "info"
    )
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logging.info("Server terminated by user.")
    except Exception as e:
        logging.critical(f"Unhandled exception in main_async: {e}", exc_info=True)

# The code now integrates FastAPI with FastMCP, providing both REST API endpoints and MCP tool functionality
