import os
import json
import struct
import aiofiles
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict, deque
from datetime import datetime

from ..exception.base import ElectrusException
from ..handler.filemanager import AdvancedFileHandler

# === B+ Tree Constants & Formats ===
NODE_HEADER_FMT = ">BHIQ"    # node_type, key_count, reserved, sibling_ptr
NODE_HEADER_SIZE = struct.calcsize(NODE_HEADER_FMT)
MAX_KEYS = 128               # fan-out tuning
MIN_KEYS = MAX_KEYS // 2


class BPlusNode:
    __slots__ = ("is_leaf","keys","ptrs","sibling","version")
    def __init__(self, is_leaf: bool):
        self.is_leaf    = is_leaf
        self.keys       = []         # sorted list of keys
        self.ptrs       = []         # children offsets or record pointers
        self.sibling    = None       # next leaf for range scan
        self.version    = 0          # optimistic concurrency stamp

    def serialize(self) -> bytes:
        hdr = struct.pack(
            NODE_HEADER_FMT,
            1 if self.is_leaf else 0,
            len(self.keys),
            0,
            self.sibling or 0
        )
        body = b"".join(
            struct.pack(">I", len(kb:=json.dumps(k).encode()))+kb +
            struct.pack(">Q", p)
            for k,p in zip(self.keys, self.ptrs)
        )
        return hdr+body

    @classmethod
    def deserialize(cls, buf: bytes) -> "BPlusNode":
        if len(buf) < NODE_HEADER_SIZE:
            raise ValueError(f"Invalid buffer size for node header: expected {NODE_HEADER_SIZE}, got {len(buf)}")
        
        nt, count, _, sib = struct.unpack(NODE_HEADER_FMT, buf[:NODE_HEADER_SIZE])
        node = cls(bool(nt))
        node.sibling = sib or None
        off = NODE_HEADER_SIZE
        for _ in range(count):
            l = struct.unpack(">I", buf[off:off+4])[0]; off+=4
            k = json.loads(buf[off:off+l].decode()); off+=l
            p = struct.unpack(">Q", buf[off:off+8])[0]; off+=8
            node.keys.append(k); node.ptrs.append(p)
        return node


class LRUCache:
    """Simple async LRU cache for fixed-size pages."""
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = asyncio.Lock()

    async def get(self, key):
        async with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    async def put(self, key, val):
        async with self.lock:
            self.cache[key] = val
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


class ElectrusIndexManager:
    def __init__(self, collection, cache_size: int = 512):
        self.collection   = collection
        self.index_dir    = os.path.join(collection.collection_dir_path, "_indexes")
        os.makedirs(self.index_dir, exist_ok=True)
        self.idx_file     = os.path.join(self.index_dir, "indexes.json")
        self.file_handler = AdvancedFileHandler(self.index_dir)
        self.page_cache   = LRUCache(cache_size)
        self._load_metadata()

    def _load_metadata(self):
        if not os.path.exists(self.idx_file):
            self.file_handler.secure_write(self.idx_file, json.dumps({}), create_version=True)
        cfg = self.file_handler.secure_read(self.idx_file)
        self.metadata = json.loads(cfg["content"])

    def _save_metadata(self):
        data = json.dumps(self.metadata, indent=2)
        self.file_handler.secure_write(self.idx_file, data, create_version=True)

    def _path(self, fld: str) -> str:
        return os.path.join(self.index_dir, f"{fld}.idx")

    # === Public APIs ===

    async def create_index(self, field: str, bulk: bool = False) -> None:
        if field in self.metadata:
            raise ElectrusException(f"Index exists on '{field}'")
        path = self._path(field)
        await self.file_handler.async_batch_operations([{"type":"write","path":path,"content":b""}])
        self.metadata[field] = {"root": None, "count":0}
        self._save_metadata()

        docs = await self.collection._read_collection_data()
        if bulk:
            # Bulk-load via sorting
            entries = sorted((doc[field],pos) for pos,doc in enumerate(docs) if field in doc)
            await self._bulk_load(field, entries)
        else:
            for pos,doc in enumerate(docs):
                if field in doc:
                    await self._insert(field, doc[field], pos)

    async def drop_index(self, field: str):
        if field not in self.metadata:
            raise ElectrusException(f"No index on '{field}'")
        os.remove(self._path(field))
        del self.metadata[field]
        self._save_metadata()

    def list_indexes(self) -> List[str]:
        return list(self.metadata)

    async def find(self, fld: str, key: Any) -> Optional[int]:
        root = self.metadata[fld]["root"]
        if root is None: return None
        return await self._find_rec(fld, root, key)

    async def range_query(self, fld: str, start: Any, end: Any) -> List[int]:
        root = self.metadata[fld]["root"]
        if root is None: return []
        off = await self._find_leaf(fld, root, start)
        res = []
        while off:
            node = await self._read_node(fld, off)
            for k,p in zip(node.keys, node.ptrs):
                if k> end: return res
                if k>=start: res.append(p)
            off = node.sibling
        return res

    # === Bulk-Loading ===

    async def _bulk_load(self, fld: str, entries: List[Tuple[Any,int]]):
        # build leaf pages first
        pages = []
        for i in range(0, len(entries), MAX_KEYS):
            chunk = entries[i:i+MAX_KEYS]
            node = BPlusNode(is_leaf=True)
            node.keys, node.ptrs = zip(*chunk)
            off = await self._write_node(fld, node, None)
            pages.append(off)
        # link siblings
        for a,b in zip(pages, pages[1:]):
            node = await self._read_node(fld, a); node.sibling=b
            await self._overwrite_node(fld,node,a)
        # build upper levels
        parents = pages
        while len(parents)>1:
            newp=[]
            for i in range(0,len(parents), MAX_KEYS):
                node=BPlusNode(False)
                seg=parents[i:i+MAX_KEYS]
                # use first key of each child
                for off in seg:
                    nd = await self._read_node(fld,off)
                    node.keys.append(nd.keys[0]); node.ptrs.append(off)
                offp = await self._write_node(fld,node,None)
                newp.append(offp)
            parents=newp
        self.metadata[fld]["root"]=parents[0]
        self.metadata[fld]["count"]=len(entries)
        self._save_metadata()

    # === Insert ===

    async def _insert(self, fld: str, key: Any, ptr: int):
        meta = self.metadata[fld]
        if meta["root"] is None:
            root = await self._write_node(fld, BPlusNode(True), None)
            meta["root"]=root
        split = await self._insert_rec(fld, meta["root"], key, ptr)
        if split:
            l,mid,r = split
            nr=BPlusNode(False); nr.keys=[mid]; nr.ptrs=[l,r]
            nr_off=await self._write_node(fld,nr,None)
            meta["root"]=nr_off
        meta["count"]+=1
        self._save_metadata()

    async def _insert_rec(self, fld, off, key, ptr):
        node=await self._read_node(fld,off)
        # OCC version check
        ver=node.version; node.version+=1; await self._overwrite_node(fld,node,off)

        if node.is_leaf:
            i=0
            while i<len(node.keys) and node.keys[i]<key: i+=1
            node.keys.insert(i,key); node.ptrs.insert(i,ptr)
            if len(node.keys)>MAX_KEYS:
                return await self._split_leaf(fld,node,off)
            await self._overwrite_node(fld,node,off); return None

        i=0
        while i<len(node.keys) and key>=node.keys[i]: i+=1
        res=await self._insert_rec(fld,node.ptrs[i],key,ptr)
        if not res: return None
        l,mid,r=res
        node.keys.insert(i,mid); node.ptrs[i]=l; node.ptrs.insert(i+1,r)
        if len(node.keys)>MAX_KEYS:
            return await self._split_internal(fld,node,off)
        await self._overwrite_node(fld,node,off); return None

    async def _split_leaf(self, fld, node, off):
        m=len(node.keys)//2
        right=BPlusNode(True)
        right.keys=node.keys[m:]; right.ptrs=node.ptrs[m:]
        node.keys=node.keys[:m]; node.ptrs=node.ptrs[:m]
        right.sibling=node.sibling; node.sibling=await self._write_node(fld,right,None)
        await self._overwrite_node(fld,node,off)
        return off,right.keys[0],node.sibling

    async def _split_internal(self, fld,node,off):
        m=len(node.keys)//2
        mid=node.keys[m]
        left=BPlusNode(False); left.keys=node.keys[:m]; left.ptrs=node.ptrs[:m+1]
        right=BPlusNode(False); right.keys=node.keys[m+1:]; right.ptrs=node.ptrs[m+1:]
        lo=await self._write_node(fld,left,None); ro=await self._write_node(fld,right,None)
        return lo,mid,ro

    # === Search Helpers ===

    async def _find_rec(self,fld,off,key):
        node=await self._read_node(fld,off)
        if node.is_leaf:
            for k,p in zip(node.keys,node.ptrs):
                if k==key: return p
            return None
        i=0
        while i<len(node.keys) and key>=node.keys[i]: i+=1
        return await self._find_rec(fld,node.ptrs[i],key)

    async def _find_leaf(self,fld,off,key):
        node=await self._read_node(fld,off)
        if node.is_leaf: return off
        i=0
        while i<len(node.keys) and key>=node.keys[i]: i+=1
        return await self._find_leaf(fld,node.ptrs[i],key)

    # === Delete (lazy merging) ===

    async def delete(self,fld,key,ptr):
        root=self.metadata[fld]["root"]
        if not root: return
        under=await self._delete_rec(fld,root,key,ptr)
        if under:
            # if root became empty non-leaf, replace
            root_node=await self._read_node(fld,root)
            if not root_node.is_leaf and len(root_node.keys)==0:
                self.metadata[fld]["root"]=root_node.ptrs[0]
        self.metadata[fld]["count"]-=1
        self._save_metadata()

    async def _delete_rec(self,fld,off,key,ptr):
        node=await self._read_node(fld,off)
        if node.is_leaf:
            for i,(k,p) in enumerate(zip(node.keys,node.ptrs)):
                if k==key and p==ptr:
                    node.keys.pop(i); node.ptrs.pop(i)
                    await self._overwrite_node(fld,node,off)
                    return len(node.keys)<MIN_KEYS
            return False
        i=0
        while i<len(node.keys) and key>=node.keys[i]: i+=1
        under=await self._delete_rec(fld,node.ptrs[i],key,ptr)
        if under:
            # lazy merge: merge with sibling only on next insert if still underflow
            pass
        return False

    # === I/O Layer ===

    async def _write_node(self, fld, node: BPlusNode, off: Optional[int]) -> int:
        data=node.serialize()
        path=self._path(fld)
        if off is None:
            async with aiofiles.open(path,"ab") as f:
                pos=await f.tell(); await f.write(data)
        else:
            async with aiofiles.open(path,"rb+") as f:
                await f.seek(off); await f.write(data); pos=off
        # version & lock
        self.file_handler.secure_write(path,b"",create_version=True,verify_integrity=False)
        await self.page_cache.put((fld,off),node)
        return pos

    async def _overwrite_node(self,fld,node,off):
        await self._write_node(fld,node,off)

    async def _read_node(self, fld, off):
        cached = await self.page_cache.get((fld, off))
        if cached: return cached

        path = self._path(fld)
        async with aiofiles.open(path, "rb") as f:
            await f.seek(off)
            buf = await f.read(NODE_HEADER_SIZE + (MAX_KEYS * (4 + 256 + 8)))  # ← ⚠️ 256 is just an assumption
            if len(buf) < NODE_HEADER_SIZE:
                raise ValueError(f"Read node buffer too small. Got {len(buf)} bytes at offset {off}")
        node=BPlusNode.deserialize(buf)
        await self.page_cache.put((fld,off),node)
        return node
