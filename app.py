#!/usr/bin/env python3
"""
AgenticCore Master v2.0 - Advanced Unified Server for AI Agents
Holistic integration of FastMCP server architecture with comprehensive tool suite
"""

import argparse
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
import datetime
from datetime import timedelta
import functools
import hashlib
import json
import logging
import logging
import mimetypes
import os
from pathlib import Path
import pickle
import re
import sqlite3
import subprocess
import threading
import time
import traceback
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)
import aiohttp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import yaml

# FastMCP imports with proper error handling
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
    
    # Application context for lifecycle management
    @dataclass
    class AppContext:
        memory: Optional['MemoryEngine'] = None
        llm: Optional['LLMInterface'] = None
        workspace: Optional['WorkspaceManager'] = None
        
    app_context = AppContext()
    
    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
        """Manage application lifecycle with type-safe context."""
        logger.info("Starting application lifecycle management...")
        
        # Initialize core components
        memory = MemoryEngine(config.memory_db_path)
        llm = LLMInterface()
        workspace = WorkspaceManager(config.workspace_path)
        
        # Initialize HTTP session for LLM
        llm.session = aiohttp.ClientSession()
        
        # Detect and set Ollama model
        global OLLAMA_MODEL_NAME
        OLLAMA_MODEL_NAME = _detect_and_set_ollama_model_sync()
        
        # Discover available models
        await llm._discover_models()
        
        # Update app context
        app_context.memory = memory
        app_context.llm = llm
        app_context.workspace = workspace
        
        try:
            yield app_context
        finally:
            logger.info("Shutting down application resources...")
            
            # Cleanup resources
            if llm.session and not llm.session.closed:
                await llm.session.close()
                
            if memory.conn:
                memory.conn.close()
                
            logger.info("Application resources cleaned up successfully.")
    
    # Create FastMCP instance with lifespan management
    mcp = FastMCP(
        name="AgenticCoreServer",
        version="2.0.0",
        lifespan=app_lifespan
    )
    
except ImportError as e:
    # Fallback for systems without FastMCP
    logger.warning(f"FastMCP not available: {e}")
    
    class FastMCP:
        def __init__(self, name: str, version: str = "1.0.0", description: str = "", lifespan=None):
            self.name = name
            self.version = version
            self.description = description
            self._tools = {}
            logging.error(f"FastMCP not available - running in fallback mode")
        
        def tool(self, name: str = "", description: str = ""):
            def decorator(func):
                self._tools[name or func.__name__] = func
                return func
            return decorator
        
        def run(self, transport: str = None, port: int = None):
            logging.error("FastMCP not available - cannot start server")
            return
    
    @dataclass
    class AppContext:
        memory: Optional['MemoryEngine'] = None
        llm: Optional['LLMInterface'] = None
        workspace: Optional['WorkspaceManager'] = None
    
    app_context = AppContext()
    MCP_AVAILABLE = False
    mcp = FastMCP("AgenticCoreServer", "2.0.0")

@dataclass
class RuntimeConfig:
    """Unified runtime configuration for the system"""
    # Ollama Configuration
    ollama_api_url: str = "http://localhost:11434"
    ollama_keep_alive: int = 0
    preferred_model_size: str = "small"
    model_warmup_enabled: bool = False
    
    # Memory System Configuration
    memory_db_path: Path = Path.home() / '.agentic_memory.db'
    max_memory_entries: int = 50000
    memory_cleanup_threshold: float = 0.85
    semantic_threshold: float = 0.75
    compression_level: int = 6
    
    # Performance Configuration
    llm_timeout: int = 180
    max_concurrent_llm_calls: int = 5
    cache_ttl_seconds: int = 7200
    context_window_size: int = 8192
    
    # Workspace Configuration
    workspace_path: Path = Path.home() / 'agentic_workspace'
    max_file_size_mb: int = 10
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.md', '.json', '.yaml', '.yml', '.py', '.js', '.html', '.css', '.xml', '.csv',
        '.ts', '.jsx', '.tsx', '.go', '.rs', '.cpp', '.c', '.h', '.java', '.kt', '.swift'
    ])
    
    # Security Configuration
    safe_mode_enabled: bool = True
    max_command_timeout: int = 300
    blocked_commands: List[str] = field(default_factory=lambda: [
        'rm -rf', 'del', 'format', 'mkfs', 'dd if=', 'chmod 777',
        'sudo rm', '> /dev/', 'curl | sh', 'wget | sh'
    ])
    
    # Server Configuration
    server_port: int = 8501
    server_host: str = "127.0.0.1"

def load_config(config_path: str = "config.yaml") -> RuntimeConfig:
    """Load configuration from YAML file with fallback to defaults"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert paths to Path objects
            if 'memory_db_path' in config_data:
                config_data['memory_db_path'] = Path(config_data['memory_db_path'])
            if 'workspace_path' in config_data:
                config_data['workspace_path'] = Path(config_data['workspace_path'])
            
            return RuntimeConfig(**config_data)
        else:
            logging.warning(f"Config file {config_path} not found, using default configuration")
            return RuntimeConfig()
    except Exception as e:
        logging.error(f"Error loading config: {e}. Using defaults.")
        return RuntimeConfig()

config = load_config()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.home() / '.agentic_core.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Global state management
_AVAILABLE_OLLAMA_MODELS = []
OLLAMA_MODEL_NAME = None

class SecurityValidator:
    """Enhanced security validation for command execution"""
    
    BLOCKED_COMMANDS = {
        'rm -rf', 'del', 'format', 'mkfs', 'dd if=', 'chmod 777',
        'sudo rm', '> /dev/', 'curl | sh', 'wget | sh', 'eval',
        'exec', '__import__', 'open(', 'file(', 'input()',
        'raw_input()', 'compile(', 'reload('
    }
    
    BLOCKED_PATTERNS = [
        r'rm\s+-rf\s+/',
        r'>\s*/dev/',
        r'\|\s*sh',
        r'sudo\s+rm',
        r'chmod\s+777',
    ]
    
    @classmethod
    def validate_command_safety(cls, command: str) -> tuple[bool, str]:
        """Validate command safety before execution"""
        for blocked in cls.BLOCKED_COMMANDS:
            if blocked in command.lower():
                return False, f"Blocked dangerous command: {blocked}"
        
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Blocked dangerous pattern: {pattern}"
        
        if any(char in command for char in ['`', '$(']):
            return False, "Command substitution not allowed"
        
        return True, "Command validated"

class MemoryEngine:
    """Enhanced memory engine with advanced semantic search and optimization"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            stop_words='english', 
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MemoryWorker")
        self.access_patterns = defaultdict(list)
        self._memory_lock = threading.RLock()
        self._initialize()
    
    def _initialize(self):
        """Initialize database with enhanced schema and indexes"""
        self._setup_database()
    
    def _setup_database(self):
        """Setup database with optimized schema and WAL mode"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.execute('PRAGMA cache_size=10000')
        self.conn.execute('PRAGMA temp_store=MEMORY')
        
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                category TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                embedding_vector BLOB,
                compression_ratio REAL DEFAULT 1.0,
                sentiment_score REAL,
                complexity_score REAL,
                memory_type TEXT DEFAULT 'user',
                source TEXT,
                metadata TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_key ON memories(key);
            CREATE INDEX IF NOT EXISTS idx_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_access_count ON memories(access_count);
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_composite_search ON memories(category, importance, access_count);
            
            CREATE TABLE IF NOT EXISTS memory_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id INTEGER,
                target_memory_id INTEGER,
                relationship_type TEXT,
                strength REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_memory_id) REFERENCES memories(id),
                FOREIGN KEY (target_memory_id) REFERENCES memories(id)
            );
        ''')
    
    def save(self, key: str, value: str, category: str = "", importance: float = 0.5, 
             tags: List[str] = None, auto_analyze: bool = True, memory_type: str = "user",
             source: str = "", metadata: Dict[str, Any] = None) -> bool:
        """Enhanced save with metadata and relationship tracking"""
        if self.conn is None:
            logger.error("Database connection not initialized")
            return False
        
        tags = tags or []
        metadata = metadata or {}
        
        with self._memory_lock:
            try:
                embedding = self._compute_embedding(value)
                sentiment = self._analyze_sentiment(value) if auto_analyze else 0.0
                complexity = self._analyze_complexity(value) if auto_analyze else 0.5
                
                compressed_value, compression_ratio = self._smart_compress(value)
                metadata_json = json.dumps(metadata) if metadata else None
                
                self.conn.execute('''
                    INSERT OR REPLACE INTO memories 
                    (key, value, category, importance, access_count, updated_at, last_accessed,
                     tags, embedding_vector, compression_ratio, sentiment_score, complexity_score,
                     memory_type, source, metadata)
                    VALUES (?, ?, ?, ?, COALESCE((SELECT access_count FROM memories WHERE key = ?), 0), 
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (key, compressed_value, category or "", importance, key, 
                      json.dumps(tags) if tags else None,
                      pickle.dumps(embedding) if embedding is not None else None,
                      compression_ratio, sentiment, complexity, memory_type, source, metadata_json))
                
                self.conn.commit()
                self._smart_optimize()
                return True
            except Exception as e:
                logger.error(f"Error saving memory '{key}': {e}")
                return False
    
    def retrieve(self, key: str, update_access: bool = True) -> Optional[Dict[str, Any]]:
        """Enhanced retrieve with metadata support"""
        if self.conn is None:
            logger.error("Database connection not initialized")
            return None
        
        with self._memory_lock:
            try:
                if update_access:
                    self.conn.execute('''
                        UPDATE memories 
                        SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP 
                        WHERE key = ?
                    ''', (key,))
                
                cursor = self.conn.execute('''
                    SELECT id, key, value, category, importance, access_count, created_at, 
                           updated_at, last_accessed, tags, compression_ratio,
                           sentiment_score, complexity_score, memory_type, source, metadata
                    FROM memories WHERE key = ?
                ''', (key,))
                
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['value'] = self._smart_decompress(result['value'], result['compression_ratio'])
                    result['tags'] = json.loads(result['tags']) if result['tags'] else []
                    result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                    
                    self.access_patterns[key].append(time.time())
                    return result
                return None
            except Exception as e:
                logger.error(f"Error retrieving memory '{key}': {e}")
                return None
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Enhanced embedding computation with Ollama integration"""
        try:
            global OLLAMA_MODEL_NAME 
            if OLLAMA_MODEL_NAME:
                ollama_embedding_url = f"{config.ollama_api_url}/api/embeddings"
                embedding_payload = {
                    "model": OLLAMA_MODEL_NAME,
                    "prompt": text,
                    "options": {"keep_alive": config.ollama_keep_alive}
                }
                try:
                    response = requests.post(ollama_embedding_url, json=embedding_payload, timeout=30)
                    response.raise_for_status()
                    embedding_data = response.json()
                    embedding_vector = np.array(embedding_data['embedding'])
                    logger.debug(f"Generated Ollama embedding for text (length: {len(text)})")
                    return embedding_vector
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to get Ollama embedding: {e}. Using hash fallback.")
            
            # Enhanced hash-based embedding fallback
            text_hash = hashlib.sha256(text.encode()).digest()
            hash_vector = np.frombuffer(text_hash, dtype=np.uint8).astype(np.float32)
            normalized_vector = hash_vector / np.linalg.norm(hash_vector)
            return normalized_vector
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None
    
    def _analyze_sentiment(self, text: str) -> float:
        """Enhanced sentiment analysis"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 
            'success', 'brilliant', 'outstanding', 'superb', 'magnificent', 'marvelous',
            'perfect', 'awesome', 'impressive', 'effective', 'beneficial', 'valuable'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'failed', 'negative', 
            'error', 'wrong', 'poor', 'inadequate', 'deficient', 'problematic', 'flawed',
            'disastrous', 'catastrophic', 'useless', 'worthless', 'harmful', 'dangerous'
        }
        
        words = set(re.findall(r'\b\w+\b', text.lower()))
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _analyze_complexity(self, text: str) -> float:
        """Enhanced complexity analysis"""
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        avg_sentence_length = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = len(set(words))
        total_words = len(words)
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0.0
        
        long_words = sum(1 for word in words if len(word) > 6)
        long_word_ratio = long_words / total_words if total_words > 0 else 0.0
        
        complexity = min(1.0, (
            (avg_sentence_length / 30.0) * 0.4 + 
            vocabulary_diversity * 0.3 + 
            long_word_ratio * 0.3
        ))
        return float(complexity)
    
    def _smart_compress(self, text: str) -> Tuple[bytes, float]:
        """Enhanced compression with adaptive thresholds"""
        if len(text) < 200:
            return text.encode('utf-8'), 1.0
        
        try:
            import zlib
            compressed = zlib.compress(text.encode('utf-8'), level=config.compression_level)
            compression_ratio = len(compressed) / len(text.encode('utf-8'))
            
            if compression_ratio < 0.85:
                return compressed, compression_ratio
        except Exception as e:
            logger.warning(f"Compression failed: {e}. Storing uncompressed.")
        
        return text.encode('utf-8'), 1.0
    
    def _smart_decompress(self, data: bytes, compression_ratio: float) -> str:
        """Enhanced decompression with error handling"""
        if compression_ratio >= 1.0:
            return data.decode('utf-8')
        
        try:
            import zlib
            return zlib.decompress(data).decode('utf-8')
        except Exception as e:
            logger.error(f"Decompression failed: {e}. Returning raw data.")
            return data.decode('utf-8', errors='ignore')
    
    def semantic_search(self, query: str, limit: int = 20, min_similarity: float = 0.0,
                       category_filter: str = "", memory_type_filter: str = "") -> List[Dict[str, Any]]:
        """Enhanced semantic search with additional filters"""
        if self.conn is None:
            logger.error("Database connection not initialized")
            return []
        
        with self._memory_lock:
            try:
                query_embedding = self._compute_embedding(query)
                if query_embedding is None:
                    logger.warning("Query embedding could not be computed for semantic search.")
                    return []
                
                sql = '''
                    SELECT id, key, value, category, importance, access_count, created_at, 
                           updated_at, last_accessed, tags, compression_ratio,
                           sentiment_score, complexity_score, memory_type, source, metadata, embedding_vector
                    FROM memories WHERE embedding_vector IS NOT NULL
                '''
                params = []
                
                if category_filter:
                    sql += ' AND category = ?'
                    params.append(category_filter)
                
                if memory_type_filter:
                    sql += ' AND memory_type = ?'
                    params.append(memory_type_filter)
                
                cursor = self.conn.execute(sql, params)
                
                results = []
                for row in cursor:
                    try:
                        stored_embedding_blob = row['embedding_vector']
                        if stored_embedding_blob is None:
                            continue

                        stored_embedding = pickle.loads(stored_embedding_blob)
                        
                        if stored_embedding.ndim > 1:
                            stored_embedding = stored_embedding.flatten()
                        if query_embedding.ndim > 1:
                            query_embedding_flat = query_embedding.flatten()
                        else:
                            query_embedding_flat = query_embedding

                        if stored_embedding.shape != query_embedding_flat.shape:
                            logger.debug(f"Embedding dimension mismatch. Stored: {stored_embedding.shape}, Query: {query_embedding_flat.shape}. Skipping.")
                            continue

                        similarity = cosine_similarity([query_embedding_flat], [stored_embedding])[0][0]
                        
                        threshold = min_similarity or config.semantic_threshold
                        if similarity >= threshold:
                            result = dict(row)
                            result['value'] = self._smart_decompress(
                                result['value'], result['compression_ratio']
                            )
                            result['tags'] = json.loads(result['tags']) if result['tags'] else []
                            result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                            result['similarity_score'] = float(similarity)
                            del result['embedding_vector']
                            results.append(result)
                    except Exception as inner_e:
                        logger.warning(f"Error processing semantic search row: {inner_e}")
                        continue
                
                # Enhanced ranking algorithm
                def ranking_score(item):
                    sim_score = item.get('similarity_score', 0.0)
                    importance = item.get('importance', 0.5)
                    access_count = item.get('access_count', 0)
                    complexity_score = item.get('complexity_score', 0.5)
                    recency_bonus = self._calculate_recency_bonus(item.get('last_accessed', ''))

                    return (sim_score * 0.35 +
                            importance * 0.25 +
                            min(access_count / 100, 1.0) * 0.20 +
                            complexity_score * 0.10 +
                            recency_bonus * 0.10)
                
                results.sort(key=ranking_score, reverse=True)
                return results[:limit]
            except Exception as e:
                logger.error(f"Error in semantic search: {e}")
                return []
    
    def _calculate_recency_bonus(self, last_accessed: str) -> float:
        """Calculate recency bonus for search ranking"""
        try:
            if not last_accessed:
                return 0.0
            
            from datetime import datetime
            last_access_time = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
            time_diff = datetime.now() - last_access_time
            days_ago = time_diff.total_seconds() / (24 * 3600)
            
            return max(0.0, np.exp(-days_ago / 30))
        except Exception:
            return 0.0
    
    def _smart_optimize(self):
        """Enhanced optimization with relationship cleanup"""
        if self.conn is None:
            logger.error("Database connection not initialized")
            return
        
        try:
            cursor = self.conn.execute('SELECT COUNT(*) as count FROM memories')
            row = cursor.fetchone()
            if row is None:
                return
            count = row['count']
            
            if count > config.max_memory_entries * config.memory_cleanup_threshold:
                cleanup_query = '''
                    DELETE FROM memories WHERE id IN (
                        SELECT m.id FROM memories m
                        LEFT JOIN memory_relationships mr ON (m.id = mr.source_memory_id OR m.id = mr.target_memory_id)
                        GROUP BY m.id
                        ORDER BY (
                            m.importance * 0.3 + 
                            (m.access_count / 100.0) * 0.25 +
                            (julianday('now') - julianday(m.last_accessed)) * -0.15 +
                            COALESCE(m.complexity_score, 0.5) * 0.15 +
                            COALESCE(COUNT(mr.id) / 10.0, 0) * 0.15
                        ) ASC 
                        LIMIT ?
                    )
                '''
                cleanup_count = int(count * (1 - config.memory_cleanup_threshold))
                if cleanup_count > 0:
                    self.conn.execute(cleanup_query, (cleanup_count,))
                    self.conn.execute('''
                        DELETE FROM memory_relationships 
                        WHERE source_memory_id NOT IN (SELECT id FROM memories)
                           OR target_memory_id NOT IN (SELECT id FROM memories)
                    ''')
                    self.conn.commit()
                    logger.info(f"Optimized memory: removed {cleanup_count} entries and orphaned relationships")
        except Exception as e:
            logger.error(f"Error during optimization: {e}")

class LLMInterface:
    """Enhanced LLM interface with advanced model management"""
    
    def __init__(self):
        self.cache = {}
        self.cache_metadata = {}
        self.models = {}
        self.performance_history = defaultdict(list)
        self.current_model = None
        self.session = None
        self.model_semaphore = asyncio.Semaphore(config.max_concurrent_llm_calls)

    async def _discover_models(self):
        """Enhanced model discovery with health checking"""
        try:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()

            response = await self.session.get(f"{config.ollama_api_url}/api/tags", timeout=10)
            async with response:
                response.raise_for_status()
                data = await response.json()
                models_list = data.get('models', [])
                
                new_models = {}
                for model_info in models_list:
                    model_name = model_info['name']
                    model_size = model_info.get('size', 0)
                    
                    new_models[model_name] = {
                        'name': model_name,
                        'size': model_size,
                        'size_category': self._categorize_model_size(model_size),
                        'performance_score': self.models.get(model_name, {}).get('performance_score', 0.5),
                        'is_preferred': self._is_preferred_model(model_name, model_size)
                    }
                
                self.models = new_models
                
                # Select preferred model
                preferred_models = [m for m in self.models.values() if m['is_preferred']]
                if preferred_models:
                    self.current_model = min(preferred_models, key=lambda x: x['size'])['name']
                else:
                    self.current_model = next(iter(self.models.keys())) if self.models else None
                
                logger.info(f"Discovered {len(self.models)} models. Selected: {self.current_model}")
                
                global OLLAMA_MODEL_NAME
                OLLAMA_MODEL_NAME = self.current_model
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
    
    def _categorize_model_size(self, size_bytes: int) -> str:
        """Categorize model by size"""
        gb_size = size_bytes / (1024 ** 3)
        if gb_size < 2:
            return "tiny"
        elif gb_size < 4:
            return "small"
        elif gb_size < 8:
            return "medium"
        elif gb_size < 15:
            return "large"
        else:
            return "huge"
    
    def _is_preferred_model(self, model_name: str, model_size: int) -> bool:
        """Check if model matches size preference"""
        size_category = self._categorize_model_size(model_size)
        
        if config.preferred_model_size == "small":
            return size_category in ["tiny", "small"]
        elif config.preferred_model_size == "medium":
            return size_category in ["small", "medium"]
        elif config.preferred_model_size == "large":
            return size_category in ["medium", "large"]
        else:
            return True
    
    async def call_async(self, prompt: str, model: Optional[str] = None, format_json: bool = False, 
                        task_type: str = "", max_retries: int = 3, temperature: float = 0.7,
                        max_tokens: Optional[int] = None, use_chat: bool = False) -> str:
        """Enhanced async LLM call"""
        selected_model = model or self.current_model
        if not selected_model:
            error_response = {"error": "No model available for LLM call."}
            logger.error(error_response["error"])
            return json.dumps(error_response, indent=2)
        
        async with self.model_semaphore:
            for attempt in range(max_retries):
                start_time = time.time()
                try:
                    enhanced_prompt = self._enhance_prompt(prompt, task_type, format_json)
                    
                    if use_chat:
                        response = await self._call_model_chat(
                            selected_model, enhanced_prompt, config.llm_timeout, temperature, max_tokens
                        )
                    else:
                        response = await self._call_model_generate(
                            selected_model, enhanced_prompt, config.llm_timeout, temperature, max_tokens
                        )
                    
                    if format_json:
                        response = self._ensure_json_format(response)
                    
                    duration = time.time() - start_time
                    self.update_performance(selected_model, duration, True)
                    
                    return response
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self.update_performance(selected_model, duration, False)
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        logger.warning(f"LLM call attempt {attempt + 1} failed: {e}. Retrying...")
                    else:
                        error_response = {
                            "error": str(e),
                            "model": selected_model,
                            "attempt": attempt + 1,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                        return json.dumps(error_response, indent=2)
        
        return json.dumps({"error": "LLM call failed unexpectedly"}, indent=2)
    
    async def _call_model_generate(self, model_name: str, prompt: str, timeout: int = 180, 
                                 temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Call using /generate endpoint"""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": max_tokens or min(4096, config.context_window_size),
                "keep_alive": config.ollama_keep_alive
            }
        }
        
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        try:
            response = await self.session.post(
                f"{config.ollama_api_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            )
            async with response:
                response.raise_for_status()
                data = await response.json()
                return data.get("response", "").strip()
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM call to {model_name} timed out after {timeout} seconds.")
        except Exception as e:
            raise e
    
    async def _call_model_chat(self, model_name: str, prompt: str, timeout: int = 180, 
                             temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Call using /chat endpoint"""
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": max_tokens or min(4096, config.context_window_size),
                "keep_alive": config.ollama_keep_alive
            }
        }
        
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        try:
            response = await self.session.post(
                f"{config.ollama_api_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            )
            async with response:
                response.raise_for_status()
                data = await response.json()
                return data.get("message", {}).get("content", "").strip()
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM chat call to {model_name} timed out after {timeout} seconds.")
        except Exception as e:
            raise e
    
    def _enhance_prompt(self, prompt: str, task_type: str, format_json: bool) -> str:
        """Enhanced prompt engineering"""
        enhancements = {
            'reasoning': "Think step by step. Show your reasoning process clearly.",
            'creative': "Be creative and think outside the box.",
            'analysis': "Provide thorough analysis with evidence and examples.",
            'fast': "Provide a concise, direct response.",
            'coding': "Generate clean, well-commented code.",
            'conversation': "Respond naturally and conversationally."
        }
        
        enhanced = prompt
        
        if task_type in enhancements:
            enhanced = f"{enhancements[task_type]}\n\n{prompt}"
        
        if format_json:
            enhanced += "\n\nPlease format your response as valid JSON."
        
        return enhanced
    
    def _ensure_json_format(self, response: str) -> str:
        """Enhanced JSON formatting"""
        try:
            parsed = json.loads(response)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            json_patterns = [
                r'\{.*\}',
                r'\[.*\]',
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```'
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group()
                        parsed = json.loads(json_str)
                        return json.dumps(parsed, indent=2)
                    except json.JSONDecodeError:
                        continue
            
            return json.dumps({
                "original_response_not_json": response,
                "note": "Original LLM response was not valid JSON",
                "timestamp": datetime.datetime.now().isoformat()
            }, indent=2)
    
    def update_performance(self, model_name: str, duration: float, success: bool):
        """Enhanced performance tracking"""
        if model_name in self.models:
            history = self.performance_history[model_name]
            history.append({
                'duration': duration, 
                'success': success, 
                'timestamp': time.time()
            })
            
            cutoff_time = time.time() - 3600
            history[:] = [h for h in history if h['timestamp'] > cutoff_time]
            
            if history:
                success_rate = sum(1 for h in history if h['success']) / len(history)
                avg_duration = sum(h['duration'] for h in history) / len(history)
                performance_score = success_rate * (1 / (1 + avg_duration / 10))
                self.models[model_name]['performance_score'] = performance_score
    
    async def close_session(self):
        """Enhanced session cleanup"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("aiohttp ClientSession closed.")

class WorkspaceManager:
    """Advanced file and workspace management"""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.workspace_path.mkdir(exist_ok=True)
        self.temp_dir = self.workspace_path / 'temp'
        self.temp_dir.mkdir(exist_ok=True)
    
    def safe_path(self, relative_path: str) -> Path:
        """Ensure path is within workspace bounds"""
        full_path = (self.workspace_path / relative_path).resolve()
        if not str(full_path).startswith(str(self.workspace_path.resolve())):
            raise ValueError(f"Path outside workspace: {relative_path}")
        return full_path
    
    def is_allowed_file(self, file_path: Path) -> bool:
        """Check if file extension is allowed"""
        return file_path.suffix.lower() in config.allowed_file_extensions
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information"""
        try:
            stat = file_path.stat()
            return {
                "path": str(file_path.relative_to(self.workspace_path)),
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "extension": file_path.suffix,
                "mime_type": mimetypes.guess_type(str(file_path))[0],
                "allowed": self.is_allowed_file(file_path)
            }
        except Exception as e:
            return {"error": str(e), "path": str(file_path)}

class SafeEditVersionControl:
    """Version control for file edits"""
    
    def __init__(self, max_versions=20):
        self.versions = {}
        self.max_versions = max_versions
        self.edit_metadata = {}
    
    def track_file(self, file_path: Path):
        """Initialize tracking for a file"""
        if file_path not in self.versions:
            self.versions[file_path] = []
            self.edit_metadata[file_path] = {
                'created_at': time.time(),
                'total_edits': 0,
                'last_edit': None
            }
    
    def create_version(self, file_path: Path, content: str, edit_type: str = "unknown"):
        """Create a new version with metadata"""
        self.track_file(file_path)
        
        version = {
            'content': content,
            'checksum': hashlib.md5(content.encode()).hexdigest(),
            'timestamp': time.time(),
            'edit_type': edit_type,
            'size': len(content)
        }
        
        self.versions[file_path].append(version)
        self.edit_metadata[file_path]['total_edits'] += 1
        self.edit_metadata[file_path]['last_edit'] = version['timestamp']
        
        if len(self.versions[file_path]) > self.max_versions:
            self.versions[file_path] = self.versions[file_path][-self.max_versions:]
    
    def get_file_info(self, file_path: Path) -> dict:
        """Get comprehensive file information"""
        if file_path not in self.versions:
            return {"tracked": False}
        
        versions = self.versions[file_path]
        metadata = self.edit_metadata[file_path]
        
        return {
            "tracked": True,
            "version_count": len(versions),
            "total_edits": metadata['total_edits'],
            "last_edit": metadata['last_edit'],
            "current_checksum": versions[-1]['checksum'] if versions else None,
            "file_size": versions[-1]['size'] if versions else 0
        }

def json_response(data, status="ok", metadata: Dict[str, Any] = None):
    """Enhanced JSON response with metadata support"""
    response = {"status": status}
    
    if metadata:
        response["metadata"] = metadata
    
    if isinstance(data, dict):
        response.update(data)
    else:
        response["result"] = data
    
    response["timestamp"] = datetime.datetime.now().isoformat()
    return json.dumps(response, indent=2, default=str)

def tool_error_handler(func):
    """Enhanced error handler with detailed error information"""
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_info = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "function": func.__name__,
                    "traceback": traceback.format_exc()
                }
                logger.error(f"Tool error in {func.__name__}: {e}")
                return json_response(error_info, status="error")
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "function": func.__name__,
                    "traceback": traceback.format_exc()
                }
                logger.error(f"Tool error in {func.__name__}: {e}")
                return json_response(error_info, status="error")
        return sync_wrapper

# Pydantic models for FastMCP 2.0 tools
class MemoryInput(BaseModel):
    action: Literal["save", "retrieve", "search", "batch_save", "batch_retrieve", "delete", "list_categories", "stats"] = Field(..., description="Operation to perform")
    key: str = Field("", description="Memory key for save/retrieve operations")
    value: str = Field("", description="Content to store in memory")
    query: str = Field("", description="Search query for search operations")
    category: str = Field("", description="Category filter for organization")
    importance: float = Field(0.5, description="Importance score (0.0-1.0)", ge=0.0, le=1.0)
    tags: Optional[List[str]] = Field(None, description="List of tags for categorization")
    auto_analyze: bool = Field(True, description="Whether to automatically analyze sentiment/complexity")
    search_type: Literal["semantic", "keyword", "category"] = Field("semantic", description="Search method")
    limit: int = Field(10, description="Maximum number of results to return", ge=1, le=100)
    min_similarity: float = Field(0.3, description="Minimum similarity threshold", ge=0.0, le=1.0)
    memory_type: str = Field("user", description="Type of memory")
    source: str = Field("", description="Source of the memory entry")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    batch_items: Optional[List[Dict[str, Any]]] = Field(None, description="List of items for batch operations")
    keys: Optional[List[str]] = Field(None, description="List of keys for batch retrieval")
    update_access: bool = Field(True, description="Whether to update access count")

class CommandInput(BaseModel):
    command: str = Field(..., description="Shell command to execute with safety validations")
    shell_type: Literal["bash", "sh", "zsh"] = Field("bash", description="Shell environment to use for execution")
    timeout: int = Field(30, description="Maximum execution time in seconds (1-300)", ge=1, le=300)
    working_directory: Optional[str] = Field(None, description="Working directory for command execution")

class LLMInput(BaseModel):
    action: Literal["generate", "chat", "analyze", "list_models", "model_stats"] = Field(..., description="Operation type")
    prompt: str = Field("", description="Input prompt for generation")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: float = Field(0.7, description="Generation temperature (0.0-2.0)", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate", ge=1)
    format_json: bool = Field(False, description="Force JSON output format")
    task_type: str = Field("", description="Task type hint")
    use_chat_endpoint: bool = Field(False, description="Use /chat endpoint instead of /generate")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation messages")
    system_message: str = Field("", description="System message for chat endpoint")

# MCP tool definitions using Pydantic models
@mcp.tool()
@tool_error_handler
def memory_system(input: MemoryInput) -> str:
    """Comprehensive memory management system with save, retrieve, search, and relationship management"""
    
    if input.action == "save":
        if not input.key or not input.value:
            return json_response({"error": "Both 'key' and 'value' are required for save operation"}, status="error")
        
        success = app_context.memory.save(
            input.key, input.value, input.category, input.importance, 
            input.tags or [], input.auto_analyze, input.memory_type, input.source, input.metadata or {}
        )
        return json_response({"saved": success, "key": input.key})
    
    elif input.action == "retrieve":
        if not input.key:
            return json_response({"error": "Key is required for retrieve operation"}, status="error")
        
        result = app_context.memory.retrieve(input.key, input.update_access)
        if not result:
            return json_response({"error": "Memory not found", "key": input.key}, status="error")
        
        serializable_result = {k: v for k, v in result.items() if k != "embedding"}
        return json_response(serializable_result)
    
    elif input.action == "search":
        if not input.query:
            return json_response({"error": "Query is required for search operation"}, status="error")
        
        if input.search_type == "semantic":
            results = app_context.memory.semantic_search(input.query, input.limit, input.min_similarity, input.category, input.memory_type)
        elif input.search_type == "keyword":
            if app_context.memory.conn is None:
                return json_response({"error": "Database connection not initialized"}, status="error")
            
            cursor = app_context.memory.conn.execute(
                """SELECT id, key, value, category, importance, access_count, created_at, 
                   updated_at, last_accessed, tags, compression_ratio, sentiment_score, 
                   complexity_score, memory_type, source, metadata 
                   FROM memories WHERE (value LIKE ? OR key LIKE ?) 
                   AND (? = '' OR category = ?) 
                   AND (? = '' OR memory_type = ?)
                   ORDER BY importance DESC, access_count DESC LIMIT ?""",
                (f"%{input.query}%", f"%{input.query}%", input.category, input.category, input.memory_type, input.memory_type, input.limit)
            )
            results = []
            for row in cursor:
                result_dict = dict(row)
                result_dict['value'] = app_context.memory._smart_decompress(
                    result_dict['value'], result_dict['compression_ratio']
                )
                result_dict['tags'] = json.loads(result_dict['tags']) if result_dict['tags'] else []
                result_dict['metadata'] = json.loads(result_dict['metadata']) if result_dict['metadata'] else {}
                results.append(result_dict)
        elif input.search_type == "category":
            if app_context.memory.conn is None:
                return json_response({"error": "Database connection not initialized"}, status="error")
            
            cursor = app_context.memory.conn.execute(
                """SELECT id, key, value, category, importance, access_count, created_at, 
                   updated_at, last_accessed, tags, compression_ratio, sentiment_score, 
                   complexity_score, memory_type, source, metadata 
                   FROM memories WHERE category = ? ORDER BY importance DESC, access_count DESC LIMIT ?""",
                (input.category, input.limit)
            )
            results = []
            for row in cursor:
                result_dict = dict(row)
                result_dict['value'] = app_context.memory._smart_decompress(
                    result_dict['value'], result_dict['compression_ratio']
                )
                result_dict['tags'] = json.loads(result_dict['tags']) if result_dict['tags'] else []
                result_dict['metadata'] = json.loads(result_dict['metadata']) if result_dict['metadata'] else {}
                results.append(result_dict)
        else:
            return json_response({"error": f"Unknown search type: {input.search_type}"}, status="error")
        
        serializable_results = []
        for r in results:
            s_r = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            serializable_results.append(s_r)
        
        return json_response({
            "query": input.query,
            "search_type": input.search_type,
            "result_count": len(serializable_results),
            "results": serializable_results
        })
    
    elif input.action == "batch_save":
        if not input.batch_items:
            return json_response({"error": "batch_items list is required for batch_save operation"}, status="error")
        
        results = []
        for item in input.batch_items:
            k = item.get("key", "")
            v = item.get("value", "")
            c = item.get("category", "")
            imp = item.get("importance", 0.5)
            t = item.get("tags", [])
            a = item.get("auto_analyze", True)
            mt = item.get("memory_type", "user")
            s = item.get("source", "")
            m = item.get("metadata", {})
            
            if not k or not v:
                results.append({"key": k, "saved": False, "error": "Missing key or value"})
                continue
            
            success = app_context.memory.save(k, v, c, imp, t, a, mt, s, m)
            results.append({"key": k, "saved": success})
        
        return json_response({"batch_save_results": results})
    
    elif input.action == "batch_retrieve":
        if not input.keys:
            return json_response({"error": "keys list is required for batch_retrieve operation"}, status="error")
        
        batch_results = []
        for k in input.keys:
            result = app_context.memory.retrieve(k, input.update_access)
            if result:
                serializable_result = {kk: vv for kk, vv in result.items() if kk != "embedding"}
                batch_results.append(serializable_result)
            else:
                batch_results.append({"error": "Not found", "key": k})
        
        return json_response({"batch_retrieve_results": batch_results})
    
    elif input.action == "stats":
        if app_context.memory.conn is None:
            return json_response({"error": "Database connection not initialized"}, status="error")
        
        cursor = app_context.memory.conn.execute("SELECT COUNT(*) as total FROM memories")
        total = cursor.fetchone()['total']
        
        cursor = app_context.memory.conn.execute("SELECT category, COUNT(*) as count FROM memories GROUP BY category")
        categories = {row['category'] or 'uncategorized': row['count'] for row in cursor}
        
        cursor = app_context.memory.conn.execute("SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type")
        types = {row['memory_type']: row['count'] for row in cursor}
        
        cursor = app_context.memory.conn.execute("SELECT AVG(importance) as avg_importance, AVG(access_count) as avg_access FROM memories")
        avg_stats = cursor.fetchone()
        
        return json_response({
            "total_memories": total,
            "categories": categories,
            "memory_types": types,
            "average_importance": avg_stats['avg_importance'],
            "average_access_count": avg_stats['avg_access']
        })
    
    elif input.action == "list_categories":
        if app_context.memory.conn is None:
            return json_response({"error": "Database connection not initialized"}, status="error")
        
        cursor = app_context.memory.conn.execute(
            "SELECT category, COUNT(*) as count FROM memories GROUP BY category ORDER BY count DESC"
        )
        categories = [{"category": row['category'] or 'uncategorized', "count": row['count']} for row in cursor]
        
        return json_response({"categories": categories})
    
    else:
        return json_response({"error": f"Unknown action: {input.action}"}, status="error")

@mcp.tool()
@tool_error_handler
async def execute_shell_command(input: CommandInput) -> str:
    """Execute shell commands with comprehensive safety validations and session management"""
    
    # Validate command safety
    is_safe, safety_msg = SecurityValidator.validate_command_safety(input.command)
    if not is_safe:
        return json_response({
            "command": input.command,
            "output": "",
            "error": f"Security violation: {safety_msg}",
            "returncode": -1,
            "success": False,
            "blocked": True
        }, status="error")
    
    # Set working directory
    cwd = input.working_directory if input.working_directory else os.getcwd()
    
    try:
        logger.info(f"Executing command: {input.command[:100]}...")
        
        result = subprocess.run(
            input.command,
            shell=True,
            executable=f"/bin/{input.shell_type}",
            capture_output=True,
            text=True,
            timeout=input.timeout,
            cwd=cwd,
            env={**os.environ, "TERM": "xterm-256color"}
        )
        
        return json_response({
            "command": input.command,
            "output": result.stdout[:10000],
            "error": result.stderr[:5000],
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "working_directory": cwd
        })
        
    except subprocess.TimeoutExpired:
        return json_response({
            "command": input.command,
            "output": "",
            "error": f"Command timed out after {input.timeout} seconds",
            "returncode": -1,
            "success": False,
            "timeout": True
        }, status="error")
    except Exception as e:
        logger.error(f"Command execution failed: {str(e)}")
        return json_response({
            "command": input.command,
            "output": "",
            "error": f"Execution failed: {str(e)}",
            "returncode": -1,
            "success": False
        }, status="error")

@mcp.tool()
@tool_error_handler
async def llm_operations(input: LLMInput) -> str:
    """Advanced LLM operations with flexible model selection, endpoint choice, and intelligent caching"""
    
    if input.action == "generate":
        if not input.prompt:
            return json_response({"error": "Prompt is required for generation"}, status="error")
        
        try:
            result = await app_context.llm.call_async(
                prompt=input.prompt,
                model=input.model,
                format_json=input.format_json,
                task_type=input.task_type,
                temperature=input.temperature,
                max_tokens=input.max_tokens,
                use_chat=input.use_chat_endpoint
            )
            
            return json_response({
                "generated_text": result,
                "model_used": input.model or app_context.llm.current_model,
                "parameters": {
                    "temperature": input.temperature,
                    "max_tokens": input.max_tokens,
                    "format_json": input.format_json,
                    "task_type": input.task_type,
                    "endpoint": "chat" if input.use_chat_endpoint else "generate"
                }
            })
        except Exception as e:
            return json_response({"error": f"Generation failed: {e}"}, status="error")
    
    elif input.action == "chat":
        if not input.prompt and not input.conversation_history:
            return json_response({"error": "Either prompt or conversation_history is required for chat"}, status="error")
        
        try:
            if input.conversation_history:
                messages = []
                if input.system_message:
                    messages.append({"role": "system", "content": input.system_message})
                
                for msg in input.conversation_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if content:
                        messages.append({"role": role, "content": content})
                
                if input.prompt:
                    messages.append({"role": "user", "content": input.prompt})
                
                conversation_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            else:
                conversation_prompt = input.prompt
            
            result = await app_context.llm.call_async(
                prompt=conversation_prompt,
                model=input.model,
                format_json=input.format_json,
                task_type=input.task_type or "conversation",
                temperature=input.temperature,
                max_tokens=input.max_tokens,
                use_chat=True
            )
            
            return json_response({
                "response": result,
                "model_used": input.model or app_context.llm.current_model,
                "conversation_length": len(input.conversation_history) if input.conversation_history else 0,
                "parameters": {
                    "temperature": input.temperature,
                    "max_tokens": input.max_tokens,
                    "format_json": input.format_json
                }
            })
        except Exception as e:
            return json_response({"error": f"Chat failed: {e}"}, status="error")
    
    elif input.action == "analyze":
        if not input.prompt:
            return json_response({"error": "Prompt is required for analysis"}, status="error")
        
        analysis_prompt = f"""Analyze the following content in detail:

{input.prompt}

Provide a comprehensive analysis including:
- Key themes and concepts
- Complexity assessment
- Potential improvements or insights
- Summary of main points

Format your response as structured analysis."""
        
        try:
            result = await app_context.llm.call_async(
                prompt=analysis_prompt,
                model=input.model,
                format_json=input.format_json,
                task_type="analysis",
                temperature=0.3,
                max_tokens=input.max_tokens
            )
            
            return json_response({
                "analysis": result,
                "analyzed_content_length": len(input.prompt),
                "model_used": input.model or app_context.llm.current_model
            })
        except Exception as e:
            return json_response({"error": f"Analysis failed: {e}"}, status="error")
    
    elif input.action == "list_models":
        return json_response({
            "current_model": app_context.llm.current_model,
            "available_models": list(app_context.llm.models.keys()),
            "model_details": {
                name: {
                    "size_category": info.get("size_category"),
                    "performance_score": info.get("performance_score"),
                    "is_preferred": info.get("is_preferred")
                }
                for name, info in app_context.llm.models.items()
            }
        })
    
    elif input.action == "model_stats":
        stats = {}
        for model_name, model_info in app_context.llm.models.items():
            history = app_context.llm.performance_history.get(model_name, [])
            recent_history = [h for h in history if time.time() - h['timestamp'] < 3600]
            
            stats[model_name] = {
                "total_calls": len(history),
                "recent_calls_1h": len(recent_history),
                "success_rate": (sum(1 for h in history if h['success']) / len(history)) if history else 0,
                "avg_duration": (sum(h['duration'] for h in history) / len(history)) if history else 0,
                "performance_score": model_info.get("performance_score", 0),
                "size_category": model_info.get("size_category", "unknown")
            }
        
        return json_response({
            "model_statistics": stats,
            "cache_info": {
                "cache_entries": len(app_context.llm.cache),
                "cache_size_estimate_mb": sum(len(str(v).encode()) for v in app_context.llm.cache.values()) / (1024 * 1024)
            }
        })
    
    else:
        return json_response({"error": f"Unknown action: {input.action}"}, status="error")

@mcp.tool()
@tool_error_handler
async def system_status() -> str:
    """Get comprehensive system status and health information"""
    
    # Get model information
    model_info = {}
    if app_context.llm.models:
        for model_name, model_data in app_context.llm.models.items():
            performance_history = app_context.llm.performance_history.get(model_name, [])
            recent_calls = [h for h in performance_history if time.time() - h['timestamp'] < 3600]
            
            model_info[model_name] = {
                "size_category": model_data.get('size_category', 'unknown'),
                "size_bytes": model_data.get('size', 0),
                "performance_score": model_data.get('performance_score', 0.5),
                "is_current": model_name == app_context.llm.current_model,
                "recent_calls": len(recent_calls),
                "success_rate": sum(1 for h in recent_calls if h['success']) / len(recent_calls) if recent_calls else 0,
                "avg_response_time": sum(h['duration'] for h in recent_calls) / len(recent_calls) if recent_calls else 0
            }
    
    # Get memory statistics
    memory_stats = {}
    if app_context.memory and app_context.memory.conn:
        try:
            cursor = app_context.memory.conn.execute("SELECT COUNT(*) as total FROM memories")
            total_memories = cursor.fetchone()['total']
            
            cursor = app_context.memory.conn.execute("SELECT category, COUNT(*) as count FROM memories GROUP BY category")
            category_counts = {row['category'] or 'uncategorized': row['count'] for row in cursor}
            
            cursor = app_context.memory.conn.execute("SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type")
            type_counts = {row['memory_type']: row['count'] for row in cursor}
            
            cursor = app_context.memory.conn.execute("SELECT AVG(importance) as avg_importance FROM memories")
            avg_importance = cursor.fetchone()['avg_importance']
            
            memory_stats = {
                "total_memories": total_memories,
                "by_category": category_counts,
                "by_type": type_counts,
                "average_importance": avg_importance,
                "database_path": str(app_context.memory.db_path)
            }
        except Exception as e:
            memory_stats = {"error": str(e)}
    
    # System configuration
    system_config = {
        "ollama_url": config.ollama_api_url,
        "max_concurrent_calls": config.max_concurrent_llm_calls,
        "cache_ttl_seconds": config.cache_ttl_seconds,
        "semantic_threshold": config.semantic_threshold,
        "keep_alive": config.ollama_keep_alive,
        "preferred_model_size": config.preferred_model_size,
        "workspace_path": str(config.workspace_path),
        "safe_mode_enabled": config.safe_mode_enabled,
        "server_port": config.server_port,
        "server_host": config.server_host
    }
    
    return json_response({
        "system_status": "operational",
        "current_model": app_context.llm.current_model if app_context.llm else None,
        "available_models": list(app_context.llm.models.keys()) if app_context.llm else [],
        "model_details": model_info,
        "memory_statistics": memory_stats,
        "configuration": system_config,
        "mcp_available": MCP_AVAILABLE,
        "server_type": "FastMCP + FastAPI"
    })

def _detect_and_set_ollama_model_sync() -> Optional[str]:
    """Enhanced Ollama model detection with size preference"""
    global _AVAILABLE_OLLAMA_MODELS
    try:
        response = requests.get(f"{config.ollama_api_url}/api/tags", timeout=10)
        response.raise_for_status()
        models_data = response.json()
        
        models_list = models_data.get('models', [])
        _AVAILABLE_OLLAMA_MODELS = [m['name'] for m in models_list]
        
        logger.info(f"Detected Ollama models: {_AVAILABLE_OLLAMA_MODELS}")

        if _AVAILABLE_OLLAMA_MODELS:
            if config.preferred_model_size == "small":
                models_with_size = [(m['name'], m.get('size', 0)) for m in models_list]
                models_with_size.sort(key=lambda x: x[1])
                selected_model = models_with_size[0][0]
            else:
                selected_model = _AVAILABLE_OLLAMA_MODELS[0]
            
            logger.info(f"Selected Ollama model: {selected_model} (preference: {config.preferred_model_size})")
            return selected_model
        else:
            logger.error("No models found in Ollama server.")
            return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to Ollama server at {config.ollama_api_url}. Please ensure Ollama is running with 'ollama serve'.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Ollama models: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Ollama model detection: {e}")
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AgenticCore Master Server")
    parser.add_argument("--port", type=int, default=8501, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--transport", default="sse", choices=["stdio", "sse"], help="Transport type")
    parser.add_argument("--mode", default="hybrid", choices=["mcp", "fastapi", "hybrid"], help="Server mode")
    
    return parser.parse_args()

# Create the fastAPI server
fastapi_app = FastAPI()

# Create the mcp server
mcp = FastMCP()

@fastapi_app.get("/", response_model=bool)
def root():
    return True

@mcp.tool()
def hello_tool(name: str):
    return f"Hello {name}" 

# Combine mcp and fastapi
mcp_app = mcp.http_app(transport="streamable-http")
routes = [
    *mcp_app.routes,
    *fastapi_app.routes
]
app = FastAPI(
    routes=routes,
    lifespan=mcp_app.lifespan,
)

# 2. Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
