"""
File operation tools for Google GenAI SDK.
Provides read, write, and edit functionality with safety features.
"""

import os
import sys
import re
import codecs
import chardet
import difflib
import logging
from pathlib import Path
from typing import Dict, Optional, List, Union
from datetime import datetime

# Constants
MAX_OUTPUT_SIZE = 0.25 * 1024 * 1024  # 0.25MB in bytes
MAX_LINES_TO_READ = 2000
MAX_LINE_LENGTH = 2000

# Create logger for file tools
file_tools_logger = logging.getLogger("spatial_reasoning.file_tools")


class FileToolsState:
    """Singleton to maintain state across tool invocations."""
    
    def __init__(self):
        self.read_file_timestamps: Dict[str, float] = {}
        self.cwd = os.getcwd()
    
    def normalize_path(self, file_path: str) -> str:
        """Convert relative path to absolute path."""
        if os.path.isabs(file_path):
            return file_path
        return os.path.abspath(os.path.join(self.cwd, file_path))
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        if not os.path.exists(file_path):
            return "utf-8"
        
        try:
            with open(file_path, "rb") as f:
                raw = f.read(min(32768, os.path.getsize(file_path)))
                result = chardet.detect(raw)
                encoding = result.get("encoding", "utf-8")
                # Handle common encoding aliases
                if encoding and encoding.lower() in ["ascii", "iso-8859-1"]:
                    return "utf-8"
                return encoding or "utf-8"
        except:
            return "utf-8"
    
    def detect_line_endings(self, file_path: str) -> str:
        """Detect line endings in file (CRLF or LF)."""
        if not os.path.exists(file_path):
            return "\n"  # Default to LF
        
        try:
            with open(file_path, "rb") as f:
                content = f.read(8192)  # Read first 8KB
                if b"\r\n" in content:
                    return "\r\n"
                return "\n"
        except:
            return "\n"
    
    def find_similar_file(self, file_path: str) -> Optional[str]:
        """Find files with similar names but different extensions."""
        directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        if not os.path.exists(directory):
            return None
        
        for file in os.listdir(directory):
            if os.path.splitext(file)[0] == base_name and file != os.path.basename(file_path):
                return os.path.join(directory, file)
        
        return None
    
    def create_patch(self, original: str, updated: str, file_path: str) -> List[str]:
        """Create a unified diff patch."""
        original_lines = original.splitlines(keepends=True)
        updated_lines = updated.splitlines(keepends=True)
        
        diff = list(
            difflib.unified_diff(
                original_lines,
                updated_lines,
                fromfile=file_path,
                tofile=file_path,
                lineterm="",
            )
        )
        
        return diff


# Global state instance
_file_state = FileToolsState()


def read_file(file_path: str, offset: int = 1, limit: Optional[int] = None, log_enabled: bool = True) -> Dict[str, Union[str, int]]:
    """
    Read content from a file with line numbers.
    
    This tool reads a file and returns its content with line numbers, similar to 'cat -n'.
    It tracks when files are read to ensure they haven't been modified before editing.
    
    Args:
        file_path: Path to the file to read (absolute or relative)
        offset: Starting line number (1-based). Default is 1 (beginning of file)
        limit: Maximum number of lines to read. Default reads up to 2000 lines
        log_enabled: Whether to log this operation (default True)
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - content: File content with line numbers (if successful)
        - num_lines: Number of lines read
        - start_line: Starting line number
        - total_lines: Total lines in the file
        - file_path: The file path that was read
        - error: Error message (if status is "error")
        
    Example:
        >>> result = read_file("example.txt", offset=10, limit=20)
        >>> if result["status"] == "success":
        >>>     print(result["content"])
    """
    try:
        full_file_path = _file_state.normalize_path(file_path)
        
        if log_enabled:
            file_tools_logger.debug(f"read_file called: path={file_path}, offset={offset}, limit={limit}")
        
        # Check if file exists
        if not os.path.exists(full_file_path):
            similar_file = _file_state.find_similar_file(full_file_path)
            error_msg = "File does not exist."
            if similar_file:
                error_msg += f" Did you mean {similar_file}?"
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }
        
        # Check file size
        stats = os.stat(full_file_path)
        file_size = stats.st_size
        
        # Skip size check if offset/limit provided
        if file_size > MAX_OUTPUT_SIZE and offset is None and limit is None:
            size_kb = round(file_size / 1024)
            max_kb = round(MAX_OUTPUT_SIZE / 1024)
            return {
                "status": "error",
                "error": f"File content ({size_kb}KB) exceeds maximum allowed size ({max_kb}KB). "
                         f"Please use offset and limit parameters to read specific portions of the file.",
                "file_size": file_size,
                "file_path": file_path
            }
        
        # Update read timestamp
        _file_state.read_file_timestamps[full_file_path] = datetime.now().timestamp() * 1000
        
        # Read file with encoding detection
        encoding = _file_state.detect_encoding(full_file_path)
        try:
            with open(full_file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(full_file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        
        total_lines = len(lines)
        
        # Handle offset (convert to 0-based)
        line_offset = 0 if offset == 0 else offset - 1
        
        # Apply offset and limit
        if limit is None:
            limit = min(MAX_LINES_TO_READ, total_lines - line_offset)
        
        selected_lines = lines[line_offset : line_offset + limit]
        
        # Format with line numbers and truncate long lines
        formatted_lines = []
        for i, line in enumerate(selected_lines, start=offset):
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "...\n"
            # Format: spaces + line number + tab + content
            formatted_lines.append(f"{i:6d}\t{line}")
        
        content = "".join(formatted_lines)
        
        # Final size check
        if len(content) > MAX_OUTPUT_SIZE:
            size_kb = round(len(content) / 1024)
            max_kb = round(MAX_OUTPUT_SIZE / 1024)
            return {
                "status": "error",
                "error": f"File content ({size_kb}KB) exceeds maximum allowed size ({max_kb}KB). "
                         f"Please use offset and limit parameters to read specific portions of the file.",
                "file_path": file_path
            }
        
        if log_enabled:
            file_tools_logger.info(f"Successfully read {file_path} (lines {offset}-{offset + len(selected_lines) - 1})")
        
        return {
            "status": "success",
            "content": content,
            "num_lines": len(selected_lines),
            "start_line": offset,
            "total_lines": total_lines,
            "file_path": file_path
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error reading file: {str(e)}",
            "file_path": file_path
        }


def edit_file(file_path: str, old_string: str, new_string: str, log_enabled: bool = True) -> Dict[str, Union[str, List[str]]]:
    """
    Edit a file by replacing exact string matches.
    
    This tool performs precise string replacements in files. It requires that the file
    was previously read using read_file() and validates that exactly one match exists
    for safety. The tool preserves file encoding and line endings.
    
    Args:
        file_path: Path to the file to edit
        old_string: Exact string to find and replace (must match exactly)
        new_string: String to replace with (can be empty to delete)
        log_enabled: Whether to log this operation (default True)
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - message: Success or error message
        - patch: List of unified diff lines showing changes (if successful)
        - file_path: Path to the edited file
        - error: Error message (if status is "error")
        
    Special cases:
        - If old_string is empty, creates a new file with new_string as content
        - If new_string is empty, deletes the matched string
        - Preserves original file encoding and line endings
        
    Example:
        >>> # First read the file
        >>> read_result = read_file("config.py")
        >>> # Then edit it
        >>> result = edit_file("config.py", "DEBUG = False", "DEBUG = True")
        >>> if result["status"] == "success":
        >>>     print("File edited successfully")
    """
    try:
        if log_enabled:
            file_tools_logger.debug(f"edit_file called: path={file_path}, old_len={len(old_string)}, new_len={len(new_string)}")
        
        # Validate inputs
        if old_string == new_string:
            return {
                "status": "error",
                "error": "No changes to make: old_string and new_string are exactly the same.",
                "file_path": file_path
            }
        
        full_file_path = _file_state.normalize_path(file_path)
        
        # Handle new file creation
        if old_string == "":
            if os.path.exists(full_file_path):
                return {
                    "status": "error",
                    "error": "Cannot create new file - file already exists.",
                    "file_path": file_path
                }
            
            # Create new file
            os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
            with open(full_file_path, "w", encoding="utf-8") as f:
                f.write(new_string)
            
            # Update timestamp
            _file_state.read_file_timestamps[full_file_path] = os.stat(full_file_path).st_mtime * 1000
            
            patch = _file_state.create_patch("", new_string, file_path)
            
            if log_enabled:
                file_tools_logger.info(f"Created new file: {file_path}")
            
            return {
                "status": "success",
                "message": f"Created new file: {file_path}",
                "patch": patch,
                "file_path": file_path
            }
        
        # Check if file exists
        if not os.path.exists(full_file_path):
            similar_file = _file_state.find_similar_file(full_file_path)
            error_msg = "File does not exist."
            if similar_file:
                error_msg += f" Did you mean {similar_file}?"
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }
        
        # Check if file is a Jupyter notebook
        if full_file_path.endswith(".ipynb"):
            return {
                "status": "error",
                "error": "File is a Jupyter Notebook. Use a notebook-specific tool to edit this file.",
                "file_path": file_path
            }
        
        # Check if file was read
        read_timestamp = _file_state.read_file_timestamps.get(full_file_path)
        if not read_timestamp:
            return {
                "status": "error",
                "error": "File has not been read yet. Read it first before editing.",
                "file_path": file_path
            }
        
        # Check if file was modified since last read
        stats = os.stat(full_file_path)
        last_write_time = stats.st_mtime * 1000
        if last_write_time > read_timestamp:
            return {
                "status": "error",
                "error": "File has been modified since last read. Read it again before editing.",
                "file_path": file_path
            }
        
        # Read file content
        encoding = _file_state.detect_encoding(full_file_path)
        line_endings = _file_state.detect_line_endings(full_file_path)
        
        try:
            with open(full_file_path, "r", encoding=encoding) as f:
                original_content = f.read()
        except UnicodeDecodeError:
            with open(full_file_path, "r", encoding="utf-8", errors="replace") as f:
                original_content = f.read()
        
        # Check if old_string exists
        if old_string not in original_content:
            return {
                "status": "error",
                "error": "String to replace not found in file.",
                "file_path": file_path
            }
        
        # Check for multiple matches
        matches = original_content.count(old_string)
        if matches > 1:
            return {
                "status": "error",
                "error": f"Found {matches} matches of the string to replace. "
                         f"For safety, only one match at a time is allowed. "
                         f"Add more context to make the match unique.",
                "file_path": file_path,
                "matches": matches
            }
        
        # Perform replacement
        if new_string == "" and not old_string.endswith("\n") and (old_string + "\n") in original_content:
            # Special case: delete line
            updated_content = original_content.replace(old_string + "\n", new_string)
        else:
            updated_content = original_content.replace(old_string, new_string)
        
        # Write file with proper encoding and line endings
        with open(full_file_path, "w", encoding=encoding, newline="") as f:
            f.write(updated_content.replace("\n", line_endings))
        
        # Update timestamp
        _file_state.read_file_timestamps[full_file_path] = os.stat(full_file_path).st_mtime * 1000
        
        # Create patch
        patch = _file_state.create_patch(original_content, updated_content, file_path)
        
        if log_enabled:
            file_tools_logger.info(f"Successfully edited {file_path} ({matches} replacement{'s' if matches > 1 else ''})")
        
        return {
            "status": "success",
            "message": f"Successfully edited {file_path}",
            "patch": patch,
            "file_path": file_path
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error editing file: {str(e)}",
            "file_path": file_path
        }


def write_file(file_path: str, content: str, log_enabled: bool = True) -> Dict[str, Union[str, List[str]]]:
    """
    Write or overwrite a file with new content.
    
    This tool creates a new file or completely replaces an existing file's content.
    For existing files, it requires that the file was previously read using read_file()
    to prevent accidental overwrites. The tool preserves file encoding and line endings.
    
    Args:
        file_path: Path to the file to write
        content: Complete content to write to the file
        log_enabled: Whether to log this operation (default True)
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - message: Success or error message
        - type: "create" for new files or "update" for existing files
        - patch: Unified diff showing changes (only for updates)
        - file_path: Path to the written file
        - error: Error message (if status is "error")
        
    Notes:
        - Creates parent directories if they don't exist
        - For existing files, preserves original encoding and line endings
        - For new files, uses UTF-8 encoding and LF line endings
        
    Example:
        >>> # Write a new file
        >>> result = write_file("new_config.json", '{"debug": true}')
        >>> 
        >>> # Update existing file (must read first)
        >>> read_file("existing.py")
        >>> result = write_file("existing.py", "# New content\\nprint('hello')")
    """
    try:
        if log_enabled:
            file_tools_logger.debug(f"write_file called: path={file_path}, content_len={len(content)}")
        
        full_file_path = _file_state.normalize_path(file_path)
        dir_path = os.path.dirname(full_file_path)
        
        old_content = None
        file_type = "create"
        
        # Check if file exists
        if os.path.exists(full_file_path):
            file_type = "update"
            
            # Check if file was read
            read_timestamp = _file_state.read_file_timestamps.get(full_file_path)
            if not read_timestamp:
                return {
                    "status": "error",
                    "error": "File exists but has not been read yet. Read it first before overwriting.",
                    "file_path": file_path
                }
            
            # Check if file was modified since last read
            stats = os.stat(full_file_path)
            last_write_time = stats.st_mtime * 1000
            if last_write_time > read_timestamp:
                return {
                    "status": "error",
                    "error": "File has been modified since last read. Read it again before overwriting.",
                    "file_path": file_path
                }
            
            # Read current content for patch
            encoding = _file_state.detect_encoding(full_file_path)
            try:
                with open(full_file_path, "r", encoding=encoding) as f:
                    old_content = f.read()
            except UnicodeDecodeError:
                with open(full_file_path, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
        
        # Determine encoding and line endings
        if os.path.exists(full_file_path):
            line_endings = _file_state.detect_line_endings(full_file_path)
            encoding = _file_state.detect_encoding(full_file_path)
        else:
            line_endings = "\n"  # Default to LF
            encoding = "utf-8"
        
        # Create directory if needed
        os.makedirs(dir_path, exist_ok=True)
        
        # Write file
        with open(full_file_path, "w", encoding=encoding, newline="") as f:
            f.write(content.replace("\n", line_endings))
        
        # Update timestamp
        _file_state.read_file_timestamps[full_file_path] = os.stat(full_file_path).st_mtime * 1000
        
        # Create response
        result = {
            "status": "success",
            "message": f"Successfully {'created' if file_type == 'create' else 'updated'} {file_path}",
            "type": file_type,
            "file_path": file_path
        }
        
        # Add patch for updates
        if old_content is not None:
            patch = _file_state.create_patch(old_content, content, file_path)
            result["patch"] = patch
        
        if log_enabled:
            file_tools_logger.info(f"Successfully {'created' if file_type == 'create' else 'updated'} {file_path}")
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error writing file: {str(e)}",
            "file_path": file_path
        }


# Export the tools for GenAI SDK
__all__ = ["read_file", "edit_file", "write_file"]