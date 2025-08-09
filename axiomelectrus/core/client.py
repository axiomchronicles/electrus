"""
Enhanced Electrus Database Manager

A modern, robust database management system with comprehensive error handling,
logging, type safety, and best practices implementation.
"""

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Union, Iterator, Any
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from ..exception.base import ElectrusException
from .database import Database


class OperationResult(Enum):
    """Enumeration for operation result types."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class DatabaseOperationResult:
    """Structured result for database operations."""
    success: bool
    message: str
    result_type: OperationResult
    data: Optional[Any] = None
    error_details: Optional[str] = None


def validate_database_name(func):
    """Decorator to validate database name parameters."""
    @wraps(func)
    def wrapper(self, db_name: str, *args, **kwargs):
        if not db_name or not isinstance(db_name, str):
            raise ElectrusException("Database name must be a non-empty string")
        
        # Validate database name format (no special characters, proper length)
        if not db_name.replace('_', '').replace('-', '').isalnum():
            raise ElectrusException(
                f"Invalid database name '{db_name}'. Only alphanumeric characters, "
                "hyphens, and underscores are allowed"
            )
        
        if len(db_name) > 64:  # Reasonable limit for filesystem compatibility
            raise ElectrusException(
                f"Database name '{db_name}' is too long. Maximum 64 characters allowed"
            )
        
        return func(self, db_name, *args, **kwargs)
    return wrapper


def log_operation(operation_name: str):
    """Decorator to log database operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.logger.info(f"Starting {operation_name} operation with args: {args}")
            try:
                result = func(self, *args, **kwargs)
                self.logger.info(f"Successfully completed {operation_name} operation")
                return result
            except Exception as e:
                self.logger.error(
                    f"Failed {operation_name} operation: {str(e)}", 
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


class ElectrusManager:
    """
    Enhanced Electrus Database Manager.
    
    A comprehensive database management system that provides robust error handling,
    logging, type safety, and modern Python best practices.
    
    Attributes:
        base_path (Path): The base directory for database storage
        logger (logging.Logger): Logger instance for operation tracking
        
    Example:
        >>> manager = ElectrusManager()
        >>> result = manager.create_database("my_database")
        >>> if result.success:
        ...     print(f"Database created: {result.message}")
    """
    
    def __init__(
        self, 
        base_path: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize the Electrus Database Manager.
        
        Args:
            base_path: Custom base path for database storage. Defaults to ~/.electrus
            logger: Custom logger instance. If None, creates a new logger
            
        Raises:
            ElectrusException: If base path cannot be created or accessed
        """
        # Use pathlib for modern path handling[23][26][35]
        if base_path is None:
            self.base_path = Path.home() / '.electrus'
        else:
            self.base_path = Path(base_path).expanduser().resolve()
        
        # Setup logging with best practices[21][24][27][30]
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
        
        # Ensure base directory exists with proper error handling[3][6]
        self._initialize_base_directory()
        
        self.logger.info(f"ElectrusManager initialized with base path: {self.base_path}")

    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging with best practices.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not logger.handlers:  # Avoid duplicate handlers
            # Create formatter with timestamp and context[27][33]
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler for development
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)
            
            # File handler for persistent logging
            log_file = self.base_path / 'electrus.log'
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
            
        return logger

    def _initialize_base_directory(self) -> None:
        """
        Initialize the base directory with proper error handling.
        
        Raises:
            ElectrusException: If directory cannot be created or accessed
        """
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Verify directory is writable
            test_file = self.base_path / '.test_write'
            try:
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError) as e:
                raise ElectrusException(
                    f"Base directory '{self.base_path}' is not writable: {e}"
                )
                
        except OSError as e:
            raise ElectrusException(
                f"Cannot create or access base directory '{self.base_path}': {e}"
            )

    @contextmanager
    def _database_operation(self, operation_name: str) -> Iterator[None]:
        """
        Context manager for database operations with comprehensive error handling.
        
        Args:
            operation_name: Name of the operation for logging
            
        Yields:
            None
        """
        self.logger.debug(f"Starting {operation_name} operation")
        try:
            yield
            self.logger.debug(f"Completed {operation_name} operation successfully")
        except ElectrusException:
            # Re-raise custom exceptions without modification
            raise
        except (OSError, IOError) as e:
            # Handle filesystem errors[3][6]
            error_msg = f"Filesystem error during {operation_name}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ElectrusException(error_msg) from e
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error during {operation_name}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ElectrusException(error_msg) from e

    @log_operation("list_databases")
    def list_databases(self) -> List[str]:
        """
        List all available databases.
        
        Returns:
            List of database names
            
        Raises:
            ElectrusException: If base directory cannot be accessed
        """
        with self._database_operation("list_databases"):
            try:
                databases = [
                    path.name for path in self.base_path.iterdir() 
                    if path.is_dir() and not path.name.startswith('.')
                ]
                self.logger.debug(f"Found {len(databases)} databases: {databases}")
                return sorted(databases)  # Return sorted for consistency
            except OSError as e:
                raise ElectrusException(f"Error accessing base directory: {e}")

    @validate_database_name
    @log_operation("list_collections")
    def list_collections_in_database(self, db_name: str) -> List[str]:
        """
        List all collections in a specific database.
        
        Args:
            db_name: Name of the database
            
        Returns:
            List of collection names
            
        Raises:
            ElectrusException: If database doesn't exist or cannot be accessed
        """
        with self._database_operation("list_collections"):
            db_path = self.base_path / db_name
            
            if not db_path.exists():
                raise ElectrusException(f"Database '{db_name}' does not exist")
            
            if not db_path.is_dir():
                raise ElectrusException(f"'{db_name}' is not a valid database directory")
            
            try:
                collections = [
                    path.name for path in db_path.iterdir()
                    if path.is_dir() and not path.name.startswith('.')
                ]
                self.logger.debug(
                    f"Found {len(collections)} collections in '{db_name}': {collections}"
                )
                return sorted(collections)
            except OSError as e:
                raise ElectrusException(
                    f"Error accessing collections in database '{db_name}': {e}"
                )

    @validate_database_name
    @log_operation("create_database")
    def create_database(self, db_name: str) -> DatabaseOperationResult:
        """
        Create a new database with comprehensive error handling.
        
        Args:
            db_name: Name of the database to create
            
        Returns:
            DatabaseOperationResult with operation details
        """
        with self._database_operation("create_database"):
            db_path = self.base_path / db_name
            
            try:
                if db_path.exists():
                    if db_path.is_dir():
                        return DatabaseOperationResult(
                            success=True,
                            message=f"Database '{db_name}' already exists",
                            result_type=OperationResult.WARNING,
                            data={"path": str(db_path)}
                        )
                    else:
                        # File exists with same name
                        raise ElectrusException(
                            f"Cannot create database '{db_name}': "
                            "a file with this name already exists"
                        )
                
                db_path.mkdir(parents=True, exist_ok=True)
                
                # Create metadata file for database tracking
                metadata_file = db_path / '.metadata'
                metadata_file.write_text(
                    f"# Electrus Database: {db_name}\n"
                    f"# Created: {Path.ctime(db_path)}\n",
                    encoding='utf-8'
                )
                
                self.logger.info(f"Successfully created database '{db_name}' at {db_path}")
                
                return DatabaseOperationResult(
                    success=True,
                    message=f"Database '{db_name}' created successfully",
                    result_type=OperationResult.SUCCESS,
                    data={"path": str(db_path)}
                )
                
            except OSError as e:
                error_msg = f"Failed to create database '{db_name}': {e}"
                return DatabaseOperationResult(
                    success=False,
                    message=error_msg,
                    result_type=OperationResult.ERROR,
                    error_details=str(e)
                )

    @validate_database_name
    @log_operation("drop_database")
    def drop_database(self, db_name: str, force: bool = False) -> DatabaseOperationResult:
        """
        Drop (delete) a database with safety checks.
        
        Args:
            db_name: Name of the database to drop
            force: If True, removes non-empty databases
            
        Returns:
            DatabaseOperationResult with operation details
            
        Raises:
            ElectrusException: If database cannot be removed
        """
        with self._database_operation("drop_database"):
            db_path = self.base_path / db_name
            
            if not db_path.exists():
                raise ElectrusException(f"Database '{db_name}' does not exist")
            
            if not db_path.is_dir():
                raise ElectrusException(f"'{db_name}' is not a valid database directory")
            
            try:
                # Check if database is empty (unless force is True)
                if not force:
                    collections = list(db_path.iterdir())
                    # Filter out metadata files
                    collections = [c for c in collections if not c.name.startswith('.')]
                    if collections:
                        raise ElectrusException(
                            f"Database '{db_name}' is not empty. "
                            f"Contains {len(collections)} collections. Use force=True to override"
                        )
                
                # Remove database directory and all contents
                shutil.rmtree(db_path)
                
                self.logger.info(f"Successfully dropped database '{db_name}'")
                
                return DatabaseOperationResult(
                    success=True,
                    message=f"Database '{db_name}' dropped successfully",
                    result_type=OperationResult.SUCCESS
                )
                
            except OSError as e:
                error_msg = f"Failed to drop database '{db_name}': {e}"
                raise ElectrusException(error_msg)

    @validate_database_name
    @log_operation("rename_database")
    def rename_database(self, old_name: str, new_name: str) -> DatabaseOperationResult:
        """
        Rename a database with validation and error handling.
        
        Args:
            old_name: Current database name
            new_name: New database name
            
        Returns:
            DatabaseOperationResult with operation details
            
        Raises:
            ElectrusException: If rename operation fails
        """
        # Validate new name as well
        if not new_name or not isinstance(new_name, str):
            raise ElectrusException("New database name must be a non-empty string")
        
        with self._database_operation("rename_database"):
            old_path = self.base_path / old_name
            new_path = self.base_path / new_name
            
            if not old_path.exists():
                raise ElectrusException(f"Database '{old_name}' does not exist")
            
            if not old_path.is_dir():
                raise ElectrusException(f"'{old_name}' is not a valid database directory")
            
            if new_path.exists():
                raise ElectrusException(f"Target name '{new_name}' already exists")
            
            try:
                old_path.rename(new_path)
                
                # Update metadata if it exists
                metadata_file = new_path / '.metadata'
                if metadata_file.exists():
                    content = metadata_file.read_text(encoding='utf-8')
                    updated_content = content.replace(
                        f"# Electrus Database: {old_name}",
                        f"# Electrus Database: {new_name}"
                    )
                    metadata_file.write_text(updated_content, encoding='utf-8')
                
                self.logger.info(f"Successfully renamed database '{old_name}' to '{new_name}'")
                
                return DatabaseOperationResult(
                    success=True,
                    message=f"Database renamed from '{old_name}' to '{new_name}' successfully",
                    result_type=OperationResult.SUCCESS,
                    data={"old_path": str(old_path), "new_path": str(new_path)}
                )
                
            except OSError as e:
                error_msg = f"Failed to rename database '{old_name}' to '{new_name}': {e}"
                raise ElectrusException(error_msg)

    @validate_database_name
    def database_exists(self, db_name: str) -> bool:
        """
        Check if a database exists.
        
        Args:
            db_name: Name of the database to check
            
        Returns:
            True if database exists and is valid, False otherwise
        """
        try:
            db_path = self.base_path / db_name
            result = db_path.exists() and db_path.is_dir()
            self.logger.debug(f"Database '{db_name}' exists: {result}")
            return result
        except Exception as e:
            self.logger.warning(f"Error checking database existence for '{db_name}': {e}")
            return False

    def get_database_info(self, db_name: str) -> dict:
        """
        Get detailed information about a database.
        
        Args:
            db_name: Name of the database
            
        Returns:
            Dictionary with database information
            
        Raises:
            ElectrusException: If database doesn't exist
        """
        if not self.database_exists(db_name):
            raise ElectrusException(f"Database '{db_name}' does not exist")
        
        db_path = self.base_path / db_name
        collections = self.list_collections_in_database(db_name)
        
        try:
            stat = db_path.stat()
            return {
                "name": db_name,
                "path": str(db_path),
                "collections_count": len(collections),
                "collections": collections,
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime,
                "size_bytes": sum(
                    f.stat().st_size for f in db_path.rglob('*') if f.is_file()
                )
            }
        except OSError as e:
            raise ElectrusException(f"Error getting database info for '{db_name}': {e}")

    @validate_database_name
    def __getitem__(self, db_name: str) -> 'Database':
        """
        Get a Database instance with validation.
        
        Args:
            db_name: Name of the database
            
        Returns:
            Database instance
            
        Raises:
            ElectrusException: If database doesn't exist
        """
        
        self.logger.debug(f"Accessing database '{db_name}'")
        return Database(db_name, base_path=self.base_path, logger=self.logger)

    def __contains__(self, db_name: str) -> bool:
        """
        Check if database exists using 'in' operator.
        
        Args:
            db_name: Name of the database
            
        Returns:
            True if database exists, False otherwise
        """
        return self.database_exists(db_name)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over database names.
        
        Yields:
            Database names
        """
        return iter(self.list_databases())

    def __len__(self) -> int:
        """
        Get the number of databases.
        
        Returns:
            Number of databases
        """
        return len(self.list_databases())

    def __repr__(self) -> str:
        """
        String representation of the manager.
        
        Returns:
            String representation
        """
        return f"ElectrusManager(base_path='{self.base_path}', databases={len(self)})"

    def close(self) -> None:
        """
        Cleanup resources and close logger handlers.
        """
        self.logger.info("Closing ElectrusManager")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Maintain backward compatibility
Electrus = ElectrusManager
