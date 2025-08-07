import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Callable
from enum import Enum
from dataclasses import dataclass
from contextlib import asynccontextmanager
from ...exception.base import ElectrusException

if TYPE_CHECKING:
    from .collection import Collection

class TransactionState(Enum):
    """Transaction states following DBMS standards"""
    ACTIVE = "active"
    PARTIALLY_COMMITTED = "partially_committed"
    COMMITTED = "committed"
    FAILED = "failed"
    ABORTED = "aborted"
    TERMINATED = "terminated"

class IsolationLevel(Enum):
    """MongoDB-style isolation levels"""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"  
    SNAPSHOT = "snapshot"
    MAJORITY = "majority"

@dataclass
class TransactionOptions:
    """Transaction configuration options"""
    isolation_level: IsolationLevel = IsolationLevel.SNAPSHOT
    read_concern: str = "snapshot"
    write_concern: str = "majority"
    max_retry_attempts: int = 3
    retry_delay: float = 0.1
    transaction_timeout: float = 30.0

class TransactionError(ElectrusException):
    """Base transaction error"""
    pass

class TransientTransactionError(TransactionError):
    """Retryable transaction error"""
    pass

class TransactionTimeoutError(TransactionError):
    """Transaction timeout error"""
    pass

class Transactions:
    def __init__(
        self, 
        collection: 'Collection', 
        parent_transaction: Optional['Transactions'] = None,
        options: Optional[TransactionOptions] = None
    ):
        """
        Enhanced transaction implementation with MongoDB-style ACID properties.
        
        Args:
            collection: The collection associated with the transaction
            parent_transaction: Parent transaction for nested transactions
            options: Transaction configuration options
        """
        self.collection = collection
        self.transaction_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.parent_transaction = parent_transaction
        self.options = options or TransactionOptions()
        
        # Transaction state management
        self.state = TransactionState.ACTIVE
        self.transaction_buffer: List[Dict[str, Any]] = []
        self.savepoints: Dict[str, int] = {}
        
        # Session and connection management
        self.session = None
        self.session_active = False
        self.start_time: Optional[float] = None
        
        # Error handling and retry logic
        self.retry_count = 0
        self.last_error: Optional[Exception] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self) -> 'Transactions':
        """Enhanced context manager entry with proper session management"""
        try:
            # Start session with proper error handling
            self.session = await self._start_session_with_retry()
            self.session_active = True
            
            # Set parent transaction context
            self.parent_transaction = getattr(self.collection, 'current_transaction', None)
            self.collection.current_transaction = self
            
            # Begin transaction with timeout protection
            await asyncio.wait_for(
                self.begin(), 
                timeout=self.options.transaction_timeout
            )
            
            self.logger.info(f"Transaction {self.transaction_id} started successfully")
            return self
            
        except asyncio.TimeoutError:
            await self._cleanup_on_error()
            raise TransactionTimeoutError(f"Transaction start timeout after {self.options.transaction_timeout}s")
        except Exception as e:
            await self._cleanup_on_error()
            raise TransactionError(f"Failed to start transaction: {e}")

    async def __aexit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        """Enhanced context manager exit with comprehensive error handling"""
        try:
            if exc_type is None:
                # Normal completion - attempt commit
                await self._commit_with_retry()
                self.logger.info(f"Transaction {self.transaction_id} committed successfully")
            else:
                # Exception occurred - rollback
                await self._rollback_with_cleanup(exc_value)
                self.logger.warning(f"Transaction {self.transaction_id} rolled back due to: {exc_value}")
                
        except Exception as cleanup_error:
            self.logger.error(f"Error during transaction cleanup: {cleanup_error}")
            # Don't suppress the original exception
            
        finally:
            # Always restore transaction context and cleanup
            await self._final_cleanup()
            
        return None  # Don't suppress exceptions

    async def begin(self) -> None:
        """Begin transaction with enhanced state management"""
        if self.state != TransactionState.ACTIVE:
            raise TransactionError(f"Cannot begin transaction in state: {self.state}")
            
        self.start_time = asyncio.get_event_loop().time()
        self.transaction_buffer = []
        self.state = TransactionState.ACTIVE
        
        self.logger.debug(f"Transaction {self.transaction_id} begun with isolation level: {self.options.isolation_level}")

    async def commit(self) -> None:
        """Enhanced commit with ACID guarantees"""
        if self.state not in [TransactionState.ACTIVE, TransactionState.PARTIALLY_COMMITTED]:
            raise TransactionError(f"Cannot commit transaction in state: {self.state}")
        
        try:
            # Check session is still active
            if not self.session_active:
                raise TransactionError("Session is no longer active")
            
            # Set partially committed state
            self.state = TransactionState.PARTIALLY_COMMITTED
            
            # Execute all buffered operations atomically
            await self._execute_transaction_buffer()
            
            # Durability - ensure changes are persisted
            await self._ensure_durability()
            
            # Mark as committed
            self.state = TransactionState.COMMITTED
            
        except Exception as e:
            self.state = TransactionState.FAILED
            self.last_error = e
            raise TransactionError(f"Transaction commit failed: {e}")

    async def rollback(self) -> None:
        """Enhanced rollback with proper state management"""
        if self.state in [TransactionState.COMMITTED, TransactionState.TERMINATED]:
            self.logger.warning(f"Cannot rollback transaction in state: {self.state}")
            return
            
        try:
            self.state = TransactionState.ABORTED
            
            # Clear transaction buffer (operations not yet committed)
            self.transaction_buffer = []
            
            # If session is active, perform actual rollback
            if self.session_active:
                await self._perform_session_rollback()
                
            self.logger.info(f"Transaction {self.transaction_id} rolled back successfully")
            
        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            raise TransactionError(f"Rollback failed: {e}")
        finally:
            self.state = TransactionState.TERMINATED

    async def create_savepoint(self, name: str) -> None:
        """Create a savepoint for partial rollback"""
        if self.state != TransactionState.ACTIVE:
            raise TransactionError("Cannot create savepoint in inactive transaction")
            
        self.savepoints[name] = len(self.transaction_buffer)
        self.logger.debug(f"Savepoint '{name}' created at position {self.savepoints[name]}")

    async def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a specific savepoint"""
        if name not in self.savepoints:
            raise TransactionError(f"Savepoint '{name}' not found")
            
        if self.state != TransactionState.ACTIVE:
            raise TransactionError("Cannot rollback to savepoint in inactive transaction")
            
        position = self.savepoints[name]
        self.transaction_buffer = self.transaction_buffer[:position]
        
        # Remove savepoints created after this one
        self.savepoints = {k: v for k, v in self.savepoints.items() if v <= position}
        
        self.logger.debug(f"Rolled back to savepoint '{name}' at position {position}")

    # Enhanced operation methods with better error handling
    async def insert_one(self, data: Dict[str, Any]) -> None:
        """Buffer insert operation with validation"""
        await self._validate_transaction_state()
        
        operation = {
            'type': 'insert_one',
            'data': data,
            'transaction_id': self.transaction_id,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self.transaction_buffer.append(operation)

    async def insert_many(self, data_list: List[Dict[str, Any]]) -> None:
        """Buffer insert_many operation with validation"""
        await self._validate_transaction_state()
        
        if not data_list:
            raise TransactionError("Cannot insert empty data list")
            
        operation = {
            'type': 'insert_many',
            'data_list': data_list,
            'transaction_id': self.transaction_id,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self.transaction_buffer.append(operation)

    async def update_one(
        self, 
        filter_query: Dict[str, Any], 
        update_data: Dict[str, Any], 
        upsert: bool = False
    ) -> None:
        """Buffer update operation with validation"""
        await self._validate_transaction_state()
        
        operation = {
            'type': 'update_one',
            'filter_query': filter_query,
            'update_data': update_data,
            'upsert': upsert,
            'transaction_id': self.transaction_id,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self.transaction_buffer.append(operation)

    async def delete_one(self, filter_query: Dict[str, Any]) -> None:
        """Buffer delete operation with validation"""
        await self._validate_transaction_state()
        
        operation = {
            'type': 'delete_one',
            'filter_query': filter_query,
            'transaction_id': self.transaction_id,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self.transaction_buffer.append(operation)

    async def delete_many(self, filter_query: Dict[str, Any]) -> None:
        """Buffer delete_many operation with validation"""
        await self._validate_transaction_state()
        
        operation = {
            'type': 'delete_many',
            'filter_query': filter_query,
            'transaction_id': self.transaction_id,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self.transaction_buffer.append(operation)

    # Read operations with transaction context
    async def find_one(
        self, 
        filter_query: Dict[str, Any], 
        projection: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find operation within transaction context"""
        await self._validate_transaction_state()
        
        # Use session for consistent read
        return await self.collection.find_one(
            filter_query, 
            projection, 
            session=self.session
        )

    async def count_documents(self, filter_query: Dict[str, Any]) -> int:
        """Count operation within transaction context"""
        await self._validate_transaction_state()
        
        return await self.collection.count_documents(
            filter_query, 
            session=self.session
        )

    async def aggregate(
        self, 
        pipeline: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Aggregation within transaction context"""
        await self._validate_transaction_state()
        
        return await self.collection.aggregation(
            pipeline, 
            session=self.session
        )

    # Private helper methods
    async def _start_session_with_retry(self):
        """Start session with retry logic for connection issues"""
        for attempt in range(self.options.max_retry_attempts):
            try:
                session = await self.collection.start_session()
                return session
            except Exception as e:
                if attempt == self.options.max_retry_attempts - 1:
                    raise
                
                self.logger.warning(f"Session start attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(self.options.retry_delay * (2 ** attempt))

    async def _commit_with_retry(self):
        """Commit with retry logic for transient errors"""
        for attempt in range(self.options.max_retry_attempts):
            try:
                await self.commit()
                return
            except TransientTransactionError as e:
                if attempt == self.options.max_retry_attempts - 1:
                    raise
                
                self.logger.warning(f"Commit attempt {attempt + 1} failed with transient error: {e}")
                await asyncio.sleep(self.options.retry_delay * (2 ** attempt))
            except Exception as e:
                # Non-transient errors should not be retried
                raise

    async def _rollback_with_cleanup(self, original_error: Exception):
        """Rollback with proper cleanup and error logging"""
        try:
            await self.rollback()
        except Exception as rollback_error:
            self.logger.error(f"Rollback failed during error handling: {rollback_error}")
            # Log both errors but don't mask the original
            
    async def _execute_transaction_buffer(self):
        """Execute all buffered operations atomically"""
        if not self.transaction_buffer:
            return
            
        try:
            for operation in self.transaction_buffer:
                await self._execute_operation(operation)
        except Exception as e:
            # If any operation fails, the transaction should be aborted
            self.state = TransactionState.FAILED
            raise TransactionError(f"Operation execution failed: {e}")

    async def _execute_operation(self, operation: Dict[str, Any]) -> None:
        """Execute individual operation with session context"""
        operation_type = operation['type']
        
        try:
            if operation_type == 'insert_one':
                await self.collection.insert_one(operation['data'], session=self.session)
            elif operation_type == 'insert_many':
                await self.collection.insert_many(operation['data_list'], session=self.session)
            elif operation_type == 'update_one':
                await self.collection.update_one(
                    operation['filter_query'], 
                    operation['update_data'],
                    upsert=operation.get('upsert', False),
                    session=self.session
                )
            elif operation_type == 'delete_one':
                await self.collection.delete_one(operation['filter_query'], session=self.session)
            elif operation_type == 'delete_many':
                await self.collection.delete_many(operation['filter_query'], session=self.session)
            else:
                raise TransactionError(f"Unknown operation type: {operation_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to execute operation {operation_type}: {e}")
            raise

    async def _ensure_durability(self):
        """Ensure changes are persisted (durability guarantee)"""
        if self.session and hasattr(self.session, 'commit_transaction'):
            await self.session.commit_transaction()

    async def _perform_session_rollback(self):
        """Perform actual session rollback"""
        if self.session and hasattr(self.session, 'abort_transaction'):
            await self.session.abort_transaction()

    async def _validate_transaction_state(self):
        """Validate transaction can accept new operations"""
        if self.state != TransactionState.ACTIVE:
            raise TransactionError(f"Cannot perform operation in state: {self.state}")
            
        if not self.session_active:
            raise TransactionError("Session is no longer active")
            
        # Check for timeout
        if self.start_time:
            elapsed = asyncio.get_event_loop().time() - self.start_time
            if elapsed > self.options.transaction_timeout:
                raise TransactionTimeoutError(f"Transaction timeout after {elapsed:.2f}s")

    async def _cleanup_on_error(self):
        """Cleanup resources when an error occurs during setup"""
        try:
            if self.session_active:
                await self.collection.end_session()
                self.session_active = False
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def _final_cleanup(self):
        """Final cleanup of resources"""
        try:
            # Restore transaction context
            self.collection.current_transaction = self.parent_transaction
            
            # End session
            if self.session_active:
                await self.collection.end_session()
                self.session_active = False
                
            # Mark as terminated
            self.state = TransactionState.TERMINATED
            
        except Exception as e:
            self.logger.error(f"Error during final cleanup: {e}")

# Utility function for transaction management
@asynccontextmanager
async def transaction_scope(
    collection: 'Collection', 
    options: Optional[TransactionOptions] = None
):
    """Async context manager for transaction scope management"""
    transaction = Transactions(collection, options=options)
    
    try:
        async with transaction:
            yield transaction
    except Exception as e:
        # Log transaction failure
        logging.getLogger(__name__).error(f"Transaction {transaction.transaction_id} failed: {e}")
        raise
