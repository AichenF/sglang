"""goal12-glue P9: PDL edge census of a captured CUDA graph (diagnostics only, SGLANG_G12_GRAPH_CENSUS=<dir>).

Walks a kept cudaGraph_t with the driver API (ctypes on libcuda): every node (type, kernel name, grid/block) and every
edge with its dependency type (0 = default / full completion, 1 = programmatic) and ports (from_port 0 = default,
1 = programmatic trigger, 2 = launch completion). Written as JSON lines: one "node" record per node (in topological
capture order, i.e. node list order) and one "edge" record per edge.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os

logger = logging.getLogger(__name__)

_lib = None


class _EdgeData(ctypes.Structure):
    _fields_ = [("from_port", ctypes.c_ubyte), ("to_port", ctypes.c_ubyte), ("type", ctypes.c_ubyte),
                ("reserved", ctypes.c_ubyte * 5)]


class _KernelParamsV2(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p), ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint),
                ("gridDimZ", ctypes.c_uint), ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint),
                ("blockDimZ", ctypes.c_uint), ("sharedMemBytes", ctypes.c_uint), ("kernelParams", ctypes.c_void_p),
                ("extra", ctypes.c_void_p), ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]


def _cuda():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libcuda.so.1")
    return _lib


_NODE_TYPES = {0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "graph", 5: "empty", 6: "wait_event",
               7: "event_record", 8: "ext_sem_signal", 9: "ext_sem_wait", 10: "mem_alloc", 11: "mem_free",
               12: "batch_mem_op", 13: "conditional"}


def _kernel_info(lib, node):
    p = _KernelParamsV2()
    fn = getattr(lib, "cuGraphKernelNodeGetParams_v2", None) or lib.cuGraphKernelNodeGetParams
    rc = fn(ctypes.c_void_p(node), ctypes.byref(p))
    if rc != 0:
        return {"err": f"getparams {rc}"}
    name = ctypes.c_char_p()
    nm = None
    if p.func:
        if lib.cuFuncGetName(ctypes.byref(name), ctypes.c_void_p(p.func)) == 0 and name.value:
            nm = name.value.decode(errors="replace")
    if nm is None and p.kern:
        if lib.cuKernelGetName(ctypes.byref(name), ctypes.c_void_p(p.kern)) == 0 and name.value:
            nm = name.value.decode(errors="replace")
    return {"name": nm, "grid": [p.gridDimX, p.gridDimY, p.gridDimZ], "block": [p.blockDimX, p.blockDimY, p.blockDimZ],
            "smem": p.sharedMemBytes}


def dump_graph(graph_ptr: int, path: str, tag: str) -> None:
    lib = _cuda()
    g = ctypes.c_void_p(graph_ptr)
    n = ctypes.c_size_t(0)
    assert lib.cuGraphGetNodes(g, None, ctypes.byref(n)) == 0
    nodes = (ctypes.c_void_p * n.value)()
    assert lib.cuGraphGetNodes(g, nodes, ctypes.byref(n)) == 0
    idx = {nodes[i]: i for i in range(n.value)}
    ne = ctypes.c_size_t(0)
    get_edges = getattr(lib, "cuGraphGetEdges_v2", None)
    if get_edges is None:
        get_edges = lib.cuGraphGetEdges  # CUDA 13: the edge-data signature is the default one
    assert get_edges(g, None, None, None, ctypes.byref(ne)) == 0
    fr = (ctypes.c_void_p * max(1, ne.value))()
    to = (ctypes.c_void_p * max(1, ne.value))()
    ed = (_EdgeData * max(1, ne.value))()
    rc = get_edges(g, fr, to, ed, ctypes.byref(ne))
    assert rc == 0, rc
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for i in range(n.value):
            t = ctypes.c_int(-1)
            lib.cuGraphNodeGetType(ctypes.c_void_p(nodes[i]), ctypes.byref(t))
            rec = {"rec": "node", "tag": tag, "i": i, "type": _NODE_TYPES.get(t.value, t.value)}
            if t.value == 0:
                rec.update(_kernel_info(lib, nodes[i]))
            f.write(json.dumps(rec) + "\n")
        for k in range(ne.value):
            f.write(json.dumps({"rec": "edge", "tag": tag, "from": idx.get(fr[k], -1), "to": idx.get(to[k], -1),
                                "type": int(ed[k].type), "from_port": int(ed[k].from_port),
                                "to_port": int(ed[k].to_port)}) + "\n")
    logger.info("goal12 graph census %s: %d nodes, %d edges -> %s", tag, n.value, ne.value, path)
