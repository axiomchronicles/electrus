# 🚀 Electrus File Handling - Your Files' Best Friend!

Welcome to the most awesome file handling system in the galaxy! 🌟 This isn't your grandma's file manager - we're talking about bulletproof, atomic, versioned, locked-down file operations that would make a Swiss bank jealous.

## What Makes This Special? ✨

Think of this as a **super-powered bodyguard** for your files:
- 🔒 **File Locking** - No more "oops, two people edited the same file" disasters
- 🔐 **Integrity Checks** - Your files stay pure with checksums 
- ⚡ **Async Operations** - Lightning fast, non-blocking operations
- 📚 **Version Control** - Time travel for your files (sort of!)
- 💣 **Atomic Operations** - All-or-nothing writes (no half-baked files)
- 📦 **Smart Backups** - Because accidents happen
- 🧵 **Thread-Safe** - Multiple threads can play nice together

## Quick Start 🏃‍♂️

```python
from electrus.file_handling import AdvancedFileHandler

# Create your file superhero
handler = AdvancedFileHandler('./my_project')

# Write like a boss (with versioning and integrity checks!)
result = handler.secure_write(
    'my_precious_data.txt',
    'This content is protected by digital ninjas!',
    create_version=True,
    verify_integrity=True
)

print(f"✅ File secured! Checksum: {result['checksum']}")
```

## The Gang of File Warriors 🦸‍♀️

### 1. FileMetadata - The Bookkeeper 📋

This little guy remembers **everything** about your files:

```python
@dataclass
class FileMetadata:
    filename: str          # "my_file.txt"
    size: int              # 1024 bytes
    checksum: str          # "abc123def456..."
    checksum_algorithm: str # "sha256"
    created_at: datetime   # When it was born
    modified_at: datetime  # Last makeover
    version: int           # Version number
    permissions: str       # "644"
```

**Real Example:**
```python
# Create some metadata magic
metadata = FileMetadata(
    filename="secret_recipe.txt",
    size=2048,
    checksum="a1b2c3d4e5f6...",
    checksum_algorithm="sha256",
    created_at=datetime.now(),
    modified_at=datetime.now(),
    version=1,
    permissions="644"
)

# Convert to dictionary for JSON storage
meta_dict = metadata.to_dict()
```

### 2. ChecksumCalculator - The Truth Detector 🕵️

This detective makes sure your files haven't been tampered with:

```python
# Check if your file is still pure
checksum = ChecksumCalculator.calculate_checksum(
    'important_file.txt', 
    algorithm='sha256'
)

# Async version for speed demons
checksum = await ChecksumCalculator.calculate_checksum_async(
    'big_file.txt',
    algorithm='sha256'
)

# Verify integrity like a pro
is_legit = ChecksumCalculator.verify_integrity(
    'suspicious_file.txt',
    expected_checksum='abc123...',
    algorithm='sha256'
)
```

**Supported Algorithms:** `md5`, `sha1`, `sha256`, `sha512`, `blake2b`

### 3. FileLockManager - The Bouncer 🚪

No more file conflicts! This bouncer makes sure only one process touches a file at a time:

```python
lock_manager = FileLockManager()

# Lock it down!
with lock_manager.acquire_lock('sensitive_file.txt', 'exclusive', timeout=5.0) as lock_info:
    # Do your dangerous file operations here
    with open('sensitive_file.txt', 'w') as f:
        f.write("Secret stuff happening...")

# Check if someone else is using the file
if lock_manager.is_locked('some_file.txt'):
    print("🚫 File is busy, try again later!")
```

### 4. AtomicFileOperations - The Perfectionist 💯

This guy ensures your writes are **all-or-nothing**. No corrupted files, ever!

```python
# Atomic write - either it works completely or not at all
with AtomicFileOperations.atomic_write('critical_data.txt', 'w') as f:
    f.write("Mission critical data that MUST be perfect!")
    # If anything goes wrong, the original file stays untouched

# Async atomic write for the speed lovers
await AtomicFileOperations.atomic_write_async(
    'async_data.txt',
    'Lightning fast atomic write!',
    mode='w'
)
```

### 5. FileVersionManager - The Time Keeper ⏰

Git for your files! Keep track of every change:

```python
version_manager = FileVersionManager('./my_project')

# Create a snapshot
metadata = version_manager.create_version('important_doc.txt')
print(f"📸 Created version {metadata.version}")

# See all versions
versions = version_manager.list_versions('important_doc.txt')
for v in versions:
    print(f"Version {v.version}: {v.modified_at}")

# Time travel - restore an old version
version_manager.restore_version('important_doc.txt', version=2, target_path='./restored_doc.txt')
```

### 6. JsonFileHandler - The JSON Whisperer 🗣️

**NEW!** Special handler for JSON files with all the bells and whistles:

```python
from electrus.file_handling import JsonFileHandler

# Initialize your JSON ninja
json_handler = JsonFileHandler('./json_data')

# Write JSON like a pro
my_data = {
    "users": ["alice", "bob", "charlie"],
    "config": {"debug": True, "max_connections": 100},
    "timestamp": "2025-01-15T10:30:00"
}

result = json_handler.write(
    'config.json',
    my_data,
    create_version=True,
    verify_integrity=True
)
print(f"🎯 JSON saved! Size: {result['size']} bytes")

# Read with verification
data = json_handler.read(
    'config.json',
    verify_integrity=True,
    expected_checksum=result['checksum']
)
print(f"📖 Loaded {len(data['data'])} keys")

# Async operations for speed
await json_handler.write_async('fast_data.json', {"speed": "ludicrous"})
result = await json_handler.read_async('fast_data.json')
```

### 7. BackupManager - The Insurance Policy 🛡️

Your files' safety net:

```python
backup_manager = BackupManager('./backups')

# Create a backup of multiple files
backup_info = backup_manager.create_backup([
    'important_file1.txt',
    'critical_data.json',
    './documents/'  # Entire directory!
])

print(f"💾 Backup '{backup_info['name']}' created!")
print(f"📊 Total size: {backup_info['total_size']} bytes")
print(f"📁 Files backed up: {len(backup_info['files'])}")
```

## The Main Event - AdvancedFileHandler 🎪

The ringmaster that brings everyone together:

```python
# Initialize the superhero team
handler = AdvancedFileHandler('./my_awesome_project')

# Secure write with ALL the features
result = handler.secure_write(
    'ultra_important.txt',
    'This content is protected by digital ninjas and quantum encryption!',
    create_version=True,      # Save a backup version
    verify_integrity=True,    # Double-check it's perfect
    algorithm='sha256'        # Use strong checksums
)

# Secure read with verification
data = handler.secure_read(
    'ultra_important.txt',
    verify_integrity=True,
    expected_checksum=result['checksum']
)

# Async batch operations - do multiple things at once!
operations = [
    {'type': 'read', 'path': 'file1.txt'},
    {'type': 'write', 'path': 'file2.txt', 'content': 'New content!'},
    {'type': 'checksum', 'path': 'file3.txt'}
]

results = await handler.async_batch_operations(operations)
for result in results:
    print(f"✅ {result['path']}: {result['status']}")
```

## Real-World Examples 🌍

### Example 1: Configuration Manager

```python
# Perfect for app configuration files
class ConfigManager:
    def __init__(self, config_dir):
        self.json_handler = JsonFileHandler(config_dir)
    
    def save_config(self, app_name, config_data):
        return self.json_handler.write(
            f'{app_name}_config.json',
            config_data,
            create_version=True  # Keep config history!
        )
    
    def load_config(self, app_name):
        try:
            result = self.json_handler.read(f'{app_name}_config.json')
            return result['data']
        except FileNotFoundError:
            return {}  # Default empty config

# Usage
config_mgr = ConfigManager('./configs')
config_mgr.save_config('myapp', {
    'database_url': 'postgresql://localhost:5432/mydb',
    'debug': False,
    'cache_timeout': 3600
})
```

### Example 2: Data Processing Pipeline

```python
async def process_data_files(file_list):
    handler = AdvancedFileHandler('./data_processing')
    
    # Read all files concurrently
    read_ops = [{'type': 'read', 'path': f} for f in file_list]
    results = await handler.async_batch_operations(read_ops)
    
    processed_data = []
    for result in results:
        if result['status'] == 'success':
            # Process the data here
            processed = transform_data(result['content'])
            processed_data.append(processed)
    
    # Save processed results atomically
    for i, data in enumerate(processed_data):
        await handler.secure_write(
            f'processed_{i}.json',
            data,
            create_version=True
        )
```

### Example 3: Database Backup System

```python
def backup_database_config():
    handler = AdvancedFileHandler('./database')
    
    # Files to backup
    critical_files = [
        'database.json',
        'users.json', 
        'permissions.json',
        './schemas/'  # Entire schema directory
    ]
    
    # Create timestamped backup
    backup_info = handler.create_backup(
        critical_files,
        backup_name=f"db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"🎉 Database backup completed!")
    print(f"📦 Backup size: {backup_info['total_size']} bytes")
    
    return backup_info
```

## Performance Tips 🏎️

1. **Use async operations** for I/O heavy workloads
2. **Batch operations** when processing multiple files
3. **Choose appropriate checksum algorithms** (sha256 is good balance of speed/security)
4. **Enable versioning only when needed** (saves storage space)
5. **Use read locks** when possible (allows concurrent reads)

## Troubleshooting 🔧

**File locked error?**
```python
# Check if file is locked before trying to access
if not lock_manager.is_locked('my_file.txt'):
    # Safe to proceed
    pass
else:
    lock_info = lock_manager.get_lock_info('my_file.txt')
    print(f"File locked by PID {lock_info.pid} since {lock_info.timestamp}")
```

**Integrity check failed?**
```python
try:
    data = handler.secure_read('file.txt', verify_integrity=True, expected_checksum='abc123')
except RuntimeError as e:
    print(f"🚨 File corruption detected: {e}")
    # Restore from backup or version
    handler.restore_file_version('file.txt', version=1, target_path='file.txt')
```

**Performance issues?**
```python
# Use smaller chunk sizes for memory-constrained environments
checksum = ChecksumCalculator.calculate_checksum('big_file.txt', chunk_size=4096)

# Use async for concurrent operations
results = await handler.async_batch_operations(many_operations)
```

That's it! Your files are now protected by the digital equivalent of a fortress. Go forth and handle files like a boss! 🎉

---
*Part of the Electrus Database Project - Making data handling awesome since 2025!* ⚡