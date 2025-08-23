#!/usr/bin/env python3
"""
Advanced FastMCP Development Agent for Cursor IDE
Sophisticated file operations, parallel search, and project analysis with enhanced security
Optimized for agent interactions and coding best practices 2024-2025
"""

from fastmcp.server import FastMCP  # Changed from 'from fastmcp import FastMCP'
import subprocess
import re
import os
import tempfile
import difflib
import hashlib
import json
import shlex
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Annotated, Literal, Union, Any
from pydantic import Field, BaseModel
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import time
from dataclasses import dataclass
from contextlib import asynccontextmanager
import mmap
import fnmatch

# Configure optimized logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize advanced MCP server with enhanced capabilities
mcp = FastMCP(
    name="AdvancedCursorDevAgent",
    version="4.0.0"
)

# --- Enhanced Security and Performance Configuration ---
@dataclass
class AdvancedSecurityConfig:
    """Enhanced security configuration with performance optimizations"""
    BLOCKED_COMMANDS = {
        'rm -rf /', 'del /s', 'format c:', 'mkfs', 'dd if=/dev', 
        'chmod 777 /', 'sudo rm -rf', '> /dev/null', 'curl | sh', 
        'wget | bash', 'eval $(', '__import__', 'exec(', 'compile(',
        'os.system', 'subprocess.call', '$()', '``'
    }
    
    BLOCKED_PATTERNS = [
        r'rm\s+-rf\s+/',
        r'>\s*/dev/',
        r'\|\s*(sh|bash)\s*$',
        r'sudo\s+rm\s+-rf',
        r'chmod\s+777\s+/',
        r'eval\s*\(',
        r'exec\s*\(',
        r'\$\([^)]*\)',
        r'`[^`]*`'
    ]
    
    MAX_FILE_SIZE_MB = 100
    MAX_SEARCH_RESULTS = 10000
    DEFAULT_TIMEOUT = 30
    MAX_PARALLEL_WORKERS = min(os.cpu_count() or 4, 8)

class SearchResult(BaseModel):
    """Enhanced search result model"""
    file_path: str
    line_number: int
    content: str
    context_before: Optional[List[str]] = None
    context_after: Optional[List[str]] = None
    match_positions: List[tuple] = []

class FileEditOperation(BaseModel):
    """Advanced file edit operation model"""
    operation_type: Literal["search_replace", "line_replace", "block_replace", "append", "prepend", "insert"]
    search_pattern: Optional[str] = None
    replacement: Optional[str] = None
    line_number: Optional[int] = None
    content: str
    validate_syntax: bool = True

# --- Advanced Development CLI with Performance Optimizations ---
class AdvancedDevCLI:
    """High-performance CLI with development-focused aliases and parallel processing"""
    
    def __init__(self):
        self.aliases: Dict[str, str] = self._init_advanced_aliases()
        self.security = AdvancedSecurityConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.security.MAX_PARALLEL_WORKERS)
        
    def _init_advanced_aliases(self) -> Dict[str, str]:
        """Initialize comprehensive development aliases"""
        return {
            # File operations
            'lf': 'find . -type f -name',
            'lg': 'rg --type py --type js --type ts --type md',
            'ldir': 'find . -type d -name',
            'recent': 'find . -type f -mtime -1',
            'modified': 'find . -type f -mmin -60',
            
            # Language-specific searches
            'pyfiles': 'fd -e py | head -50',
            'jsfiles': 'fd -e js -e ts | head -50',
            'codefiles': 'fd -e py -e js -e ts -e jsx -e tsx -e vue -e svelte | head -100',
            'configs': 'fd -e json -e yaml -e yml -e toml -e ini | head -30',
            
            # Git operations
            'gitfiles': 'git ls-files | head -50',
            'gitdiff': 'git diff --name-only',
            'gitchanged': 'git diff --name-only HEAD~1',
            'gitstatus': 'git status --porcelain',
            
            # Directory analysis
            'tree': 'tree -I "__pycache__|node_modules|.git|.venv|venv|build|dist" -L 3',
            'sizes': 'du -sh * | sort -hr | head -20',
            'bigsizes': 'find . -type f -size +1M -exec du -h {} + | sort -hr | head -20',
            
            # Advanced search patterns
            'findclass': 'rg "^class\\s+\\w+" --type py',
            'findfunction': 'rg "^def\\s+\\w+" --type py',
            'findimport': 'rg "^(import|from)\\s+" --type py',
            'findtodo': 'rg -i "(todo|fixme|hack|bug):"',
            'findtest': 'rg "def\\s+test_\\w+" --type py',
            
            # Performance monitoring
            'diskusage': 'df -h',
            'meminfo': 'free -h',
            'processes': 'ps aux | head -20'
        }

    @lru_cache(maxsize=1000)
    def validate_command_safety(self, command: str) -> tuple[bool, str]:
        """Optimized security validation with caching"""
        cmd_lower = command.lower().strip()
        
        # Quick blocked command check
        for blocked in self.security.BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return False, f"Blocked command: {blocked}"
        
        # Pattern validation
        for pattern in self.security.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Blocked pattern: {pattern}"
        
        return True, "Safe"

    def expand_alias(self, command: str) -> str:
        """Enhanced alias expansion with parameter support"""
        if command.startswith('@'):
            alias_parts = command[1:].split(' ', 1)
            alias_name = alias_parts[0]
            args = alias_parts[1] if len(alias_parts) > 1 else ''
            
            if alias_name in self.aliases:
                base_cmd = self.aliases[alias_name]
                return f"{base_cmd} {args}".strip()
        
        return command

# Global enhanced CLI instance
dev_cli = AdvancedDevCLI()

# --- Advanced Parallel Search Engine ---
class AdvancedSearchEngine:
    """High-performance search engine with GREP-like capabilities"""
    
    def __init__(self):
        self.max_workers = AdvancedSecurityConfig.MAX_PARALLEL_WORKERS
        
    async def search_content_parallel(
        self,
        pattern: str,
        path: str = ".",
        file_types: List[str] = None,
        max_results: int = 1000,
        case_sensitive: bool = False,
        context_lines: int = 0,
        use_regex: bool = True,
        exclude_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """Advanced parallel content search with multiple backends"""
        
        # Try different search tools in order of preference
        search_tools = ['rg', 'ag', 'grep']
        
        for tool in search_tools:
            try:
                if tool == 'rg':
                    return await self._search_with_ripgrep(
                        pattern, path, file_types, max_results, 
                        case_sensitive, context_lines, exclude_patterns
                    )
                elif tool == 'ag':
                    return await self._search_with_silver_searcher(
                        pattern, path, file_types, max_results, case_sensitive
                    )
                else:
                    return await self._search_with_native_parallel(
                        pattern, path, file_types, max_results, 
                        case_sensitive, context_lines, use_regex
                    )
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Search tool {tool} failed: {e}")
                continue
        
        # Fallback to native search
        return await self._search_with_native_parallel(
            pattern, path, file_types, max_results, 
            case_sensitive, context_lines, use_regex
        )
    
    async def _search_with_ripgrep(
        self,
        pattern: str,
        path: str,
        file_types: List[str],
        max_results: int,
        case_sensitive: bool,
        context_lines: int,
        exclude_patterns: List[str]
    ) -> Dict[str, Any]:
        """Optimized ripgrep search"""
        cmd = ["rg", "--json", "--max-count", str(max_results)]
        
        if not case_sensitive:
            cmd.append("-i")
        
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        
        # Add file type filters
        if file_types:
            for ext in file_types:
                cmd.extend(["-t", ext])
        
        # Add exclude patterns
        if exclude_patterns:
            for exclude in exclude_patterns:
                cmd.extend(["-g", f"!{exclude}"])
        
        cmd.extend([pattern, path])
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            results = []
            for line in stdout.decode().strip().split('\n'):
                if line:
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'match':
                            results.append(self._parse_ripgrep_match(data))
                    except json.JSONDecodeError:
                        continue
            
            return {
                "matches": results,
                "count": len(results),
                "tool": "ripgrep",
                "success": True
            }
        
        raise subprocess.CalledProcessError(process.returncode, cmd)
    
    def _parse_ripgrep_match(self, match_data: Dict) -> SearchResult:
        """Parse ripgrep JSON output"""
        return SearchResult(
            file_path=match_data['data']['path']['text'],
            line_number=match_data['data']['line_number'],
            content=match_data['data']['lines']['text'],
            match_positions=[(m['start'], m['end']) for m in match_data['data']['submatches']]
        )
    
    async def _search_with_native_parallel(
        self,
        pattern: str,
        path: str,
        file_types: List[str],
        max_results: int,
        case_sensitive: bool,
        context_lines: int,
        use_regex: bool
    ) -> Dict[str, Any]:
        """High-performance native parallel search"""
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        if use_regex:
            try:
                regex_pattern = re.compile(pattern, flags)
            except re.error as e:
                return {"error": f"Invalid regex pattern: {e}", "success": False}
        else:
            regex_pattern = re.compile(re.escape(pattern), flags)
        
        # Find files to search
        files_to_search = await self._find_files_async(path, file_types)
        
        # Parallel search execution
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self._search_file_content, 
                    file_path, regex_pattern, context_lines, max_results
                ): file_path
                for file_path in files_to_search[:1000]  # Limit files to process
            }
            
            for future in as_completed(future_to_file):
                file_results = future.result()
                results.extend(file_results)
                
                if len(results) >= max_results:
                    break
        
        return {
            "matches": results[:max_results],
            "count": len(results[:max_results]),
            "total_files_searched": len(files_to_search),
            "tool": "native_parallel",
            "success": True
        }
    
    async def _find_files_async(self, path: str, file_types: List[str]) -> List[Path]:
        """Asynchronously find files to search"""
        def find_files():
            base_path = Path(path)
            if not base_path.exists():
                return []
            
            patterns = [f"*.{ext}" for ext in (file_types or ['*'])]
            files = []
            
            for pattern in patterns:
                files.extend(base_path.rglob(pattern))
            
            # Filter out common directories to ignore
            ignored_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist'}
            return [f for f in files if f.is_file() and not any(part in ignored_dirs for part in f.parts)]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, find_files)
    
    def _search_file_content(
        self, 
        file_path: Path, 
        pattern: re.Pattern, 
        context_lines: int,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search content in a single file with memory mapping for large files"""
        results = []
        
        try:
            file_size = file_path.stat().st_size
            
            # Use memory mapping for larger files
            if file_size > 1024 * 1024:  # 1MB threshold
                with open(file_path, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        content = mm.read().decode('utf-8', errors='ignore')
            else:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            lines = content.splitlines()
            
            for line_num, line in enumerate(lines, 1):
                if pattern.search(line):
                    result = {
                        "file_path": str(file_path),
                        "line_number": line_num,
                        "content": line.strip(),
                        "match_positions": [(m.start(), m.end()) for m in pattern.finditer(line)]
                    }
                    
                    if context_lines > 0:
                        start_line = max(0, line_num - context_lines - 1)
                        end_line = min(len(lines), line_num + context_lines)
                        result["context_before"] = lines[start_line:line_num-1]
                        result["context_after"] = lines[line_num:end_line]
                    
                    results.append(result)
                    
                    if len(results) >= max_results:
                        break
        
        except Exception as e:
            logger.warning(f"Failed to search {file_path}: {e}")
        
        return results

# Global search engine instance
search_engine = AdvancedSearchEngine()

# --- Enhanced Core Development Tools ---

@mcp.tool()
async def execute_shell_command(
    command: Annotated[str, Field(description="Shell command to execute (use @alias for shortcuts)")],
    timeout: Annotated[int, Field(30, description="Timeout in seconds (max 300)")] = 30,
    working_directory: Annotated[Optional[str], Field(None, description="Working directory")] = None,
    capture_env: Annotated[bool, Field(False, description="Capture environment variables")] = False
) -> Dict[str, Any]:
    """Execute shell commands with enhanced security, performance monitoring, and alias support"""
    
    # Expand aliases
    expanded_cmd = dev_cli.expand_alias(command.strip())
    
    # Security validation
    is_safe, safety_msg = dev_cli.validate_command_safety(expanded_cmd)
    if not is_safe:
        return {
            "error": f"Security violation: {safety_msg}",
            "command": expanded_cmd,
            "success": False,
            "blocked": True
        }
    
    # Limit timeout
    timeout = min(max(timeout, 1), 300)
    cwd = working_directory or os.getcwd()
    
    start_time = time.time()
    
    try:
        # Enhanced environment
        env = os.environ.copy()
        env.update({
            "TERM": "xterm-256color",
            "PYTHONUNBUFFERED": "1",
            "FORCE_COLOR": "1"
        })
        
        process = await asyncio.create_subprocess_shell(
            expanded_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise subprocess.TimeoutExpired(expanded_cmd, timeout)
        
        execution_time = time.time() - start_time
        
        result = {
            "success": process.returncode == 0,
            "command": expanded_cmd,
            "output": stdout.decode('utf-8', errors='ignore')[:16000],
            "error": stderr.decode('utf-8', errors='ignore')[:4000],
            "returncode": process.returncode,
            "execution_time": round(execution_time, 3),
            "working_directory": cwd
        }
        
        if capture_env:
            result["environment"] = dict(env)
        
        return result
        
    except subprocess.TimeoutExpired:
        return {
            "error": f"Command timed out after {timeout}s",
            "command": expanded_cmd,
            "success": False,
            "timeout": True,
            "execution_time": timeout
        }
    except Exception as e:
        return {
            "error": f"Execution failed: {str(e)}",
            "command": expanded_cmd,
            "success": False,
            "execution_time": time.time() - start_time
        }

@mcp.tool()
async def advanced_search_content(
    pattern: Annotated[str, Field(description="Search pattern (regex supported)")],
    path: Annotated[str, Field(".", description="Search path")] = ".",
    file_types: Annotated[Optional[List[str]], Field(None, description="File extensions (e.g., ['py', 'js'])")] = None,
    max_results: Annotated[int, Field(1000, description="Maximum results")] = 1000,
    case_sensitive: Annotated[bool, Field(False, description="Case sensitive search")] = False,
    context_lines: Annotated[int, Field(0, description="Context lines around matches")] = 0,
    use_regex: Annotated[bool, Field(True, description="Use regex patterns")] = True,
    exclude_patterns: Annotated[Optional[List[str]], Field(None, description="Patterns to exclude")] = None,
    parallel: Annotated[bool, Field(True, description="Use parallel processing")] = True
) -> Dict[str, Any]:
    """Advanced content search with GREP-like capabilities, parallel processing, and multiple backends"""
    
    if not pattern.strip():
        return {"error": "Search pattern cannot be empty", "success": False}
    
    # Set default file types if none provided
    if file_types is None:
        file_types = ["py", "js", "ts", "jsx", "tsx", "vue", "md", "txt", "json", "yaml", "yml"]
    
    try:
        start_time = time.time()
        
        if parallel:
            result = await search_engine.search_content_parallel(
                pattern=pattern,
                path=path,
                file_types=file_types,
                max_results=max_results,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                use_regex=use_regex,
                exclude_patterns=exclude_patterns or []
            )
        else:
            # Fallback to sequential search
            result = await search_engine._search_with_native_parallel(
                pattern, path, file_types, max_results, 
                case_sensitive, context_lines, use_regex
            )
        
        search_time = time.time() - start_time
        result["search_time"] = round(search_time, 3)
        result["pattern"] = pattern
        result["path"] = path
        
        return result
        
    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "success": False,
            "pattern": pattern,
            "path": path
        }

@mcp.tool()
async def advanced_file_operations(
    file_path: Annotated[str, Field(description="Target file path")],
    operations: Annotated[List[FileEditOperation], Field(description="List of edit operations")],
    backup: Annotated[bool, Field(True, description="Create backup before editing")] = True,
    validate_syntax: Annotated[bool, Field(True, description="Validate syntax for code files")] = True,
    atomic: Annotated[bool, Field(True, description="Atomic file operations")] = True
) -> Dict[str, Any]:
    """Advanced file editing with multiple operations, syntax validation, and atomic writes"""
    
    path = Path(file_path)
    
    try:
        # Read current content
        if path.exists():
            original_content = path.read_text(encoding='utf-8')
            original_lines = original_content.splitlines()
        else:
            original_content = ""
            original_lines = []
        
        # Create backup if requested
        backup_path = None
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + f'.bak.{int(time.time())}')
            backup_path.write_text(original_content, encoding='utf-8')
        
        # Apply operations sequentially
        current_content = original_content
        current_lines = original_lines.copy()
        operations_applied = []
        
        for i, operation in enumerate(operations):
            try:
                if operation.operation_type == "search_replace":
                    if operation.search_pattern and operation.replacement is not None:
                        if operation.search_pattern in current_content:
                            current_content = current_content.replace(
                                operation.search_pattern, 
                                operation.replacement
                            )
                            current_lines = current_content.splitlines()
                            operations_applied.append(f"Search/replace operation {i+1}")
                        else:
                            return {
                                "error": f"Search pattern not found in operation {i+1}: {operation.search_pattern}",
                                "success": False
                            }
                
                elif operation.operation_type == "line_replace":
                    if operation.line_number and 1 <= operation.line_number <= len(current_lines):
                        current_lines[operation.line_number - 1] = operation.content
                        current_content = '\n'.join(current_lines)
                        operations_applied.append(f"Line {operation.line_number} replaced")
                    else:
                        return {
                            "error": f"Invalid line number in operation {i+1}: {operation.line_number}",
                            "success": False
                        }
                
                elif operation.operation_type == "append":
                    current_content += "\n" + operation.content if current_content else operation.content
                    current_lines = current_content.splitlines()
                    operations_applied.append(f"Content appended")
                
                elif operation.operation_type == "prepend":
                    current_content = operation.content + "\n" + current_content if current_content else operation.content
                    current_lines = current_content.splitlines()
                    operations_applied.append(f"Content prepended")
                
                elif operation.operation_type == "insert":
                    if operation.line_number and 0 <= operation.line_number <= len(current_lines):
                        current_lines.insert(operation.line_number, operation.content)
                        current_content = '\n'.join(current_lines)
                        operations_applied.append(f"Content inserted at line {operation.line_number}")
                    else:
                        return {
                            "error": f"Invalid line number for insert in operation {i+1}: {operation.line_number}",
                            "success": False
                        }
                
            except Exception as e:
                return {
                    "error": f"Operation {i+1} failed: {str(e)}",
                    "success": False,
                    "operations_applied": operations_applied
                }
        
        # Syntax validation for code files
        if validate_syntax and path.suffix == '.py':
            try:
                compile(current_content, str(path), 'exec')
            except SyntaxError as e:
                return {
                    "error": f"Python syntax error after operations: {str(e)}",
                    "success": False,
                    "operations_applied": operations_applied,
                    "backup_path": str(backup_path) if backup_path else None
                }
        
        # Atomic write
        if atomic:
            temp_path = path.with_suffix(path.suffix + '.tmp')
            temp_path.write_text(current_content, encoding='utf-8')
            temp_path.rename(path)
        else:
            path.write_text(current_content, encoding='utf-8')
        
        # Calculate changes
        lines_before = len(original_lines)
        lines_after = len(current_lines)
        
        return {
            "success": True,
            "file_path": str(path),
            "operations_applied": operations_applied,
            "backup_path": str(backup_path) if backup_path else None,
            "lines_before": lines_before,
            "lines_after": lines_after,
            "lines_changed": abs(lines_after - lines_before),
            "characters_before": len(original_content),
            "characters_after": len(current_content)
        }
        
    except Exception as e:
        return {
            "error": f"File operation failed: {str(e)}",
            "success": False,
            "file_path": str(path)
        }

@mcp.tool()
async def read_file_advanced(
    file_path: Annotated[str, Field(description="File path to read")],
    lines_start: Annotated[Optional[int], Field(None, description="Start line (1-based)")] = None,
    lines_end: Annotated[Optional[int], Field(None, description="End line (1-based)")] = None,
    max_size_mb: Annotated[int, Field(100, description="Max file size in MB")] = 100,
    encoding: Annotated[str, Field("utf-8", description="File encoding")] = "utf-8",
    include_metadata: Annotated[bool, Field(False, description="Include file metadata")] = False,
    highlight_pattern: Annotated[Optional[str], Field(None, description="Pattern to highlight")] = None
) -> Dict[str, Any]:
    """Advanced file reading with metadata, encoding support, and pattern highlighting"""
    
    path = Path(file_path)
    
    if not path.exists():
        return {"error": "File not found", "success": False, "file_path": str(path)}
    
    # Check file size
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    if size_mb > max_size_mb:
        return {
            "error": f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)",
            "success": False,
            "file_path": str(path),
            "size_mb": size_mb
        }
    
    try:
        # Read file with specified encoding
        content = path.read_text(encoding=encoding, errors='replace')
        
        # Apply line range if specified
        if lines_start is not None or lines_end is not None:
            lines = content.splitlines()
            start = (lines_start - 1) if lines_start else 0
            end = lines_end if lines_end else len(lines)
            content = '\n'.join(lines[start:end])
            line_range = f"{start + 1}-{min(end, len(lines))}"
        else:
            line_range = f"1-{len(content.splitlines())}"
        
        result = {
            "success": True,
            "file_path": str(path),
            "content": content,
            "size_mb": round(size_mb, 3),
            "line_count": len(content.splitlines()),
            "character_count": len(content),
            "encoding": encoding,
            "line_range": line_range
        }
        
        # Add metadata if requested
        if include_metadata:
            stat = path.stat()
            result["metadata"] = {
                "modified_time": stat.st_mtime,
                "created_time": stat.st_ctime,
                "permissions": oct(stat.st_mode)[-3:],
                "owner_uid": stat.st_uid,
                "group_gid": stat.st_gid
            }
        
        # Add pattern highlighting
        if highlight_pattern:
            try:
                pattern = re.compile(highlight_pattern, re.IGNORECASE)
                matches = [(m.start(), m.end(), m.group()) for m in pattern.finditer(content)]
                result["highlighted_matches"] = matches[:100]  # Limit matches
                result["match_count"] = len(matches)
            except re.error as e:
                result["highlight_error"] = f"Invalid regex pattern: {e}"
        
        return result
        
    except UnicodeDecodeError as e:
        return {
            "error": f"Encoding error with {encoding}: {str(e)}",
            "success": False,
            "file_path": str(path),
            "suggested_encodings": ["latin-1", "cp1252", "utf-16"]
        }
    except Exception as e:
        return {
            "error": f"Read failed: {str(e)}",
            "success": False,
            "file_path": str(path)
        }

@mcp.tool()
async def batch_file_operations(self,
    operations: Annotated[List[Dict[str, Any]], Field(description="List of file operations")],
    parallel: Annotated[bool, Field(True, description="Execute operations in parallel")] = True,
    max_workers: Annotated[int, Field(4, description="Maximum parallel workers")] = 4
) -> Dict[str, Any]:
    """Execute multiple file operations in batch with parallel processing support"""
    
    if not operations:
        return {"error": "No operations provided", "success": False}
    
    results = []
    start_time = time.time()
    
    try:
        if parallel and len(operations) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(max_workers, len(operations))) as executor:
                future_to_op = {
                    executor.submit(self._execute_single_file_operation, op): i 
                    for i, op in enumerate(operations)
                }
                
                for future in as_completed(future_to_op):
                    op_index = future_to_op[future]
                    try:
                        result = future.result()
                        result["operation_index"] = op_index
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "operation_index": op_index,
                            "error": f"Operation failed: {str(e)}",
                            "success": False
                        })
        else:
            # Sequential execution
            for i, operation in enumerate(operations):
                result = await self._execute_single_file_operation(operation)
                result["operation_index"] = i
                results.append(result)
        
        # Sort results by operation index
        results.sort(key=lambda x: x.get("operation_index", 0))
        
        execution_time = time.time() - start_time
        successful_operations = sum(1 for r in results if r.get("success", False))
        
        return {
            "success": True,
            "total_operations": len(operations),
            "successful_operations": successful_operations,
            "failed_operations": len(operations) - successful_operations,
            "execution_time": round(execution_time, 3),
            "parallel_execution": parallel,
            "results": results
        }
        
    except Exception as e:
        return {
            "error": f"Batch operation failed: {str(e)}",
            "success": False,
            "partial_results": results
        }

async def execute_single_file_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single file operation"""
    op_type = operation.get("type")
    
    if op_type == "read":
        return await read_file_advanced(
            file_path=operation["file_path"],
            lines_start=operation.get("lines_start"),
            lines_end=operation.get("lines_end"),
            max_size_mb=operation.get("max_size_mb", 100),
            encoding=operation.get("encoding", "utf-8"),
            include_metadata=operation.get("include_metadata", False)
        )
    elif op_type == "edit":
        return await advanced_file_operations(
            file_path=operation["file_path"],
            operations=operation["operations"],
            backup=operation.get("backup", True),
            validate_syntax=operation.get("validate_syntax", True),
            atomic=operation.get("atomic", True)
        )
    else:
        return {
            "error": f"Unknown operation type: {op_type}",
            "success": False
        }

# --- Enhanced Project Analysis with Performance Metrics ---

@mcp.tool()
async def analyze_project_advanced(self,
    path: Annotated[str, Field(".", description="Project path")] = ".",
    max_depth: Annotated[int, Field(3, description="Max directory depth")] = 3,
    include_hidden: Annotated[bool, Field(False, description="Include hidden files")] = False,
    analyze_dependencies: Annotated[bool, Field(True, description="Analyze project dependencies")] = True,
    performance_metrics: Annotated[bool, Field(True, description="Include performance metrics")] = True,
    code_metrics: Annotated[bool, Field(True, description="Calculate code metrics")] = True
) -> Dict[str, Any]:
    """Advanced project analysis with dependency analysis, performance metrics, and code statistics"""
    
    project_path = Path(path).resolve()
    if not project_path.exists():
        return {"error": "Path not found", "success": False}
    
    start_time = time.time()
    
    try:
        # Project type detection with enhanced analysis
        project_info = await self._analyze_project_type_advanced(project_path)
        
        # File statistics with parallel processing
        file_stats = await self._analyze_file_statistics_parallel(
            project_path, include_hidden, max_depth
        )
        
        # Code metrics analysis
        code_stats = {}
        if code_metrics:
            code_stats = await self._analyze_code_metrics(project_path)
        
        # Dependency analysis
        dependencies = {}
        if analyze_dependencies:
            dependencies = await self._analyze_dependencies(project_path, project_info["type"])
        
        # Performance metrics
        metrics = {}
        if performance_metrics:
            metrics = {
                "analysis_time": round(time.time() - start_time, 3),
                "disk_usage_mb": round(file_stats.get("total_size_bytes", 0) / (1024 * 1024), 2),
                "average_file_size_kb": round(
                    file_stats.get("total_size_bytes", 0) / max(file_stats.get("total_files", 1), 1) / 1024, 2
                )
            }
        
        # Directory structure with enhanced details
        structure = await self._get_directory_structure_advanced(
            project_path, max_depth, include_hidden
        )
        
        return {
            "success": True,
            "project_path": str(project_path),
            "project_info": project_info,
            "file_statistics": file_stats,
            "code_metrics": code_stats,
            "dependencies": dependencies,
            "performance_metrics": metrics,
            "directory_structure": structure
        }
        
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "success": False,
            "analysis_time": round(time.time() - start_time, 3)
        }

async def analyze_project_type_advanced(self, path: Path) -> Dict[str, Any]:
    """Enhanced project type detection with configuration analysis"""
    
    indicators = {
        "package.json": {"type": "nodejs", "config_files": ["package.json", "package-lock.json", "yarn.lock"]},
        "requirements.txt": {"type": "python", "config_files": ["requirements.txt", "setup.py", "pyproject.toml"]},
        "pyproject.toml": {"type": "python", "config_files": ["pyproject.toml", "setup.cfg", "poetry.lock"]},
        "Cargo.toml": {"type": "rust", "config_files": ["Cargo.toml", "Cargo.lock"]},
        "go.mod": {"type": "go", "config_files": ["go.mod", "go.sum"]},
        "composer.json": {"type": "php", "config_files": ["composer.json", "composer.lock"]},
        "pom.xml": {"type": "java", "config_files": ["pom.xml"]},
        "build.gradle": {"type": "java", "config_files": ["build.gradle", "settings.gradle"]},
        "Dockerfile": {"type": "docker", "config_files": ["Dockerfile", "docker-compose.yml"]},
        ".gitignore": {"type": "git", "config_files": [".gitignore", ".git/config"]}
    }
    
    detected_types = []
    config_files_found = []
    
    for file, info in indicators.items():
        if (path / file).exists():
            detected_types.append(info["type"])
            config_files_found.extend([f for f in info["config_files"] if (path / f).exists()])
    
    primary_type = detected_types[0] if detected_types else "unknown"
    
    return {
        "type": primary_type,
        "detected_types": list(set(detected_types)),
        "config_files": list(set(config_files_found)),
        "is_git_repository": (path / ".git").exists(),
        "has_docker": (path / "Dockerfile").exists() or (path / "docker-compose.yml").exists()
    }

async def _analyze_file_statistics_parallel(
    self, path: Path, include_hidden: bool, max_depth: int
) -> Dict[str, Any]:
    """Parallel file statistics analysis"""
    
    def analyze_files():
        file_stats = {}
        total_files = 0
        total_size = 0
        largest_files = []
        
        for item in path.rglob("*"):
            # Depth check
            relative_path = item.relative_to(path)
            if len(relative_path.parts) > max_depth:
                continue
                
            if item.is_file():
                if not include_hidden and any(part.startswith('.') for part in item.parts):
                    continue
                
                try:
                    size = item.stat().st_size
                    ext = item.suffix.lower() or 'no_extension'
                    
                    if ext not in file_stats:
                        file_stats[ext] = {"count": 0, "total_size": 0}
                    
                    file_stats[ext]["count"] += 1
                    file_stats[ext]["total_size"] += size
                    
                    total_files += 1
                    total_size += size
                    
                    # Track largest files
                    largest_files.append((str(item), size))
                    
                except (OSError, PermissionError):
                    continue
        
        # Sort and limit largest files
        largest_files.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "file_types": file_stats,
            "largest_files": largest_files[:10]
        }
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, analyze_files)

async def _analyze_code_metrics(self, path: Path) -> Dict[str, Any]:
    """Analyze code-specific metrics"""
    
    code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs'}
    
    def count_code_metrics():
        metrics = {
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "functions": 0,
            "classes": 0,
            "files_by_language": {}
        }
        
        for item in path.rglob("*"):
            if item.is_file() and item.suffix in code_extensions:
                try:
                    content = item.read_text(encoding='utf-8', errors='ignore')
                    lines = content.splitlines()
                    
                    lang = item.suffix
                    if lang not in metrics["files_by_language"]:
                        metrics["files_by_language"][lang] = {"files": 0, "lines": 0}
                    
                    metrics["files_by_language"][lang]["files"] += 1
                    metrics["files_by_language"][lang]["lines"] += len(lines)
                    metrics["total_lines"] += len(lines)
                    
                    # Basic line classification
                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            metrics["blank_lines"] += 1
                        elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                            metrics["comment_lines"] += 1
                        else:
                            metrics["code_lines"] += 1
                    
                    # Simple function/class counting for Python
                    if item.suffix == '.py':
                        metrics["functions"] += len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
                        metrics["classes"] += len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
                    
                except (OSError, UnicodeDecodeError):
                    continue
        
        return metrics
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, count_code_metrics)

async def _analyze_dependencies(self, path: Path, project_type: str) -> Dict[str, Any]:
    """Analyze project dependencies based on project type"""
    
    dependencies = {"found": False, "files": [], "packages": []}
    
    try:
        if project_type == "python":
            # Python dependencies
            req_files = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]
            for req_file in req_files:
                req_path = path / req_file
                if req_path.exists():
                    dependencies["found"] = True
                    dependencies["files"].append(req_file)
                    
                    if req_file == "requirements.txt":
                        content = req_path.read_text()
                        packages = [line.strip().split('==')[0].split('>=')[0].split('<=')[0] 
                                  for line in content.splitlines() 
                                  if line.strip() and not line.startswith('#')]
                        dependencies["packages"].extend(packages)
        
        elif project_type == "nodejs":
            # Node.js dependencies
            package_json = path / "package.json"
            if package_json.exists():
                dependencies["found"] = True
                dependencies["files"].append("package.json")
                
                try:
                    package_data = json.loads(package_json.read_text())
                    deps = package_data.get("dependencies", {})
                    dev_deps = package_data.get("devDependencies", {})
                    dependencies["packages"] = list(deps.keys()) + list(dev_deps.keys())
                except json.JSONDecodeError:
                    pass
        
    except Exception as e:
        dependencies["error"] = str(e)
    
    return dependencies

async def _get_directory_structure_advanced(
    self, path: Path, max_depth: int, include_hidden: bool, current_depth: int = 0
) -> Dict[str, Any]:
    """Get enhanced directory structure with file counts and sizes"""
    
    if current_depth >= max_depth:
        return {"truncated": True}
    
    def analyze_directory():
        items = []
        total_size = 0
        file_count = 0
        dir_count = 0
        
        try:
            for item in sorted(path.iterdir()):
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                if item.is_dir():
                    dir_count += 1
                    dir_info = {
                        "name": item.name,
                        "type": "directory",
                        "item_count": len(list(item.iterdir())) if current_depth < max_depth - 1 else 0
                    }
                    
                    if current_depth < max_depth - 1:
                        subdir_analysis = self._get_directory_structure_advanced(
                            item, max_depth, include_hidden, current_depth + 1
                        )
                        dir_info["contents"] = subdir_analysis
                    
                    items.append(dir_info)
                    
                else:
                    file_count += 1
                    try:
                        size = item.stat().st_size
                        total_size += size
                        items.append({
                            "name": item.name,
                            "type": "file",
                            "size": size,
                            "extension": item.suffix
                        })
                    except OSError:
                        items.append({
                            "name": item.name,
                            "type": "file",
                            "size": 0,
                            "extension": item.suffix,
                            "error": "Cannot access file stats"
                        })
            
            return {
                "items": items,
                "summary": {
                    "total_size": total_size,
                    "file_count": file_count,
                    "directory_count": dir_count
                }
            }
            
        except PermissionError:
            return {"error": "Permission denied"}
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, analyze_directory)

@mcp.tool()
async def manage_aliases_advanced(
    action: Annotated[Literal["list", "add", "remove", "export", "import"], Field(description="Alias action")],
    alias_name: Annotated[Optional[str], Field(None, description="Alias name")] = None,
    command: Annotated[Optional[str], Field(None, description="Command for alias")] = None,
    category: Annotated[Optional[str], Field(None, description="Alias category")] = None,
    file_path: Annotated[Optional[str], Field(None, description="File path for import/export")] = None
) -> Dict[str, Any]:
    """Advanced alias management with categories, import/export, and persistence"""
    
    if action == "list":
        aliases_by_category = {}
        for alias, cmd in dev_cli.aliases.items():
            # Determine category based on command
            if any(search_cmd in cmd for search_cmd in ['find', 'grep', 'rg', 'ag']):
                cat = "search"
            elif any(git_cmd in cmd for git_cmd in ['git']):
                cat = "git"
            elif any(file_cmd in cmd for file_cmd in ['ls', 'du', 'tree']):
                cat = "file"
            else:
                cat = "misc"
            
            if cat not in aliases_by_category:
                aliases_by_category[cat] = {}
            aliases_by_category[cat][alias] = cmd
        
        return {
            "aliases": dev_cli.aliases,
            "aliases_by_category": aliases_by_category,
            "total_count": len(dev_cli.aliases),
            "categories": list(aliases_by_category.keys()),
            "usage": "Use @alias_name in commands"
        }
    
    if not alias_name and action in ["add", "remove"]:
        return {"error": "alias_name required", "success": False}
    
    if action == "add":
        if not command:
            return {"error": "command required for add", "success": False}
        
        # Enhanced alias name validation
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', alias_name):
            return {"error": "Invalid alias name. Use letters, numbers, underscore, and hyphen only.", "success": False}
        
        # Validate command safety
        is_safe, msg = dev_cli.validate_command_safety(command)
        if not is_safe:
            return {"error": f"Unsafe command: {msg}", "success": False}
        
        # Add category tag if provided
        if category:
            command = f"{command} # category:{category}"
        
        dev_cli.aliases[alias_name] = command
        return {
            "success": True,
            "message": f"Alias '{alias_name}' added",
            "category": category or "misc",
            "usage": f"Use @{alias_name} in commands"
        }
    
    elif action == "remove":
        if alias_name in dev_cli.aliases:
            removed = dev_cli.aliases.pop(alias_name)
            return {
                "success": True,
                "message": f"Alias '{alias_name}' removed",
                "removed_command": removed
            }
        return {"error": f"Alias '{alias_name}' not found", "success": False}
    
    elif action == "export":
        if not file_path:
            return {"error": "file_path required for export", "success": False}
        
        try:
            export_data = {
                "version": "1.0",
                "exported_at": time.time(),
                "aliases": dev_cli.aliases
            }
            
            Path(file_path).write_text(json.dumps(export_data, indent=2), encoding='utf-8')
            return {
                "success": True,
                "message": f"Aliases exported to {file_path}",
                "exported_count": len(dev_cli.aliases)
            }
        except Exception as e:
            return {"error": f"Export failed: {str(e)}", "success": False}
    
    elif action == "import":
        if not file_path:
            return {"error": "file_path required for import", "success": False}
        
        try:
            import_path = Path(file_path)
            if not import_path.exists():
                return {"error": "Import file not found", "success": False}
            
            import_data = json.loads(import_path.read_text(encoding='utf-8'))
            imported_aliases = import_data.get("aliases", {})
            
            # Validate imported aliases
            safe_aliases = {}
            unsafe_aliases = []
            
            for alias, cmd in imported_aliases.items():
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', alias):
                    is_safe, _ = dev_cli.validate_command_safety(cmd)
                    if is_safe:
                        safe_aliases[alias] = cmd
                    else:
                        unsafe_aliases.append(alias)
                else:
                    unsafe_aliases.append(alias)
            
            # Import safe aliases
            dev_cli.aliases.update(safe_aliases)
            
            return {
                "success": True,
                "message": f"Imported {len(safe_aliases)} aliases from {file_path}",
                "imported_count": len(safe_aliases),
                "skipped_unsafe": unsafe_aliases,
                "total_aliases": len(dev_cli.aliases)
            }
            
        except Exception as e:
            return {"error": f"Import failed: {str(e)}", "success": False}

@mcp.tool()
async def list_advanced_tools() -> Dict[str, Any]:
    """List all available advanced development tools with categories and performance info"""
    
    tools = [
        {
            "name": "execute_shell_command",
            "category": "shell",
            "description": "Execute shell commands with enhanced security and performance monitoring",
            "features": ["alias_support", "timeout_control", "environment_capture", "performance_metrics"]
        },
        {
            "name": "advanced_search_content",
            "category": "search",
            "description": "Advanced content search with GREP-like capabilities and parallel processing",
            "features": ["parallel_processing", "multiple_backends", "context_lines", "regex_support", "exclude_patterns"]
        },
        {
            "name": "advanced_file_operations",
            "category": "file",
            "description": "Sophisticated file editing with multiple operations and atomic writes",
            "features": ["multiple_operations", "syntax_validation", "atomic_writes", "backup_creation"]
        },
        {
            "name": "read_file_advanced",
            "category": "file",
            "description": "Advanced file reading with metadata and pattern highlighting",
            "features": ["encoding_support", "metadata_extraction", "pattern_highlighting", "line_range_selection"]
        },
        {
            "name": "batch_file_operations",
            "category": "file",
            "description": "Execute multiple file operations in parallel batches",
            "features": ["parallel_execution", "batch_processing", "operation_tracking", "error_handling"]
        },
        {
            "name": "analyze_project_advanced",
            "category": "analysis",
            "description": "Comprehensive project analysis with metrics and dependency tracking",
            "features": ["dependency_analysis", "code_metrics", "performance_metrics", "project_type_detection"]
        },
        {
            "name": "manage_aliases_advanced",
            "category": "shell",
            "description": "Advanced alias management with categories and import/export",
            "features": ["categorization", "import_export", "safety_validation", "persistence"]
        }
    ]
    
    return {
        "tools": tools,
        "aliases": dev_cli.aliases,
        "total_tools": len(tools),
        "categories": list(set(tool["category"] for tool in tools)),
        "performance_info": {
            "max_parallel_workers": dev_cli.security.MAX_PARALLEL_WORKERS,
            "max_file_size_mb": dev_cli.security.MAX_FILE_SIZE_MB,
            "default_timeout": dev_cli.security.DEFAULT_TIMEOUT
        },
        "version": "4.0.0",
        "features": [
            "parallel_processing",
            "enhanced_security",
            "performance_monitoring",
            "advanced_search",
            "batch_operations",
            "code_analysis",
            "dependency_tracking"
        ]
    }

# --- Cleanup and Performance Optimization ---
import atexit

def cleanup_resources():
    """Cleanup resources on exit"""
    if hasattr(dev_cli, 'executor'):
        dev_cli.executor.shutdown(wait=False)

atexit.register(cleanup_resources)

if __name__ == "__main__":
    logger.info("Starting Advanced FastMCP Development Agent v4.0.0")
    logger.info(f"Parallel workers: {dev_cli.security.MAX_PARALLEL_WORKERS}")
    logger.info(f"Security patterns: {len(dev_cli.security.BLOCKED_PATTERNS)}")
    logger.info(f"Available aliases: {len(dev_cli.aliases)}")
    
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        cleanup_resources()
    except Exception as e:
        logger.error(f"Server error: {e}")
        cleanup_resources()
        raise