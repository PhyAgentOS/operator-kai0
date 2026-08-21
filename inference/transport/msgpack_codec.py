"""与 kai0 服务端（openpi-client 内嵌 msgpack_numpy）字节级一致的 numpy 编解码。

线格式契约：服务端（本仓库 server/ 快照的 openpi-client）内嵌 msgpack_numpy
（packages/openpi-client/src/openpi_client/msgpack_numpy.py）使用「字典格式」——
numpy 数组打包为普通 msgpack dict：:

    {b"__ndarray__": True, b"data": <bytes>, b"dtype": <str>, b"shape": <tuple>}

解码经 object_hook 还原。这与 PyPI msgpack-numpy（ExtType 格式）线格式**不同**；
混用会导致服务端把数组解码成 0 维空数组（服务端日志 `state_shape: ()`）。
本模块按服务端字典格式实现，保证与真实服务端字节级兼容。

不支持 V/O/c dtype（与服务端一致：直接报错，不做 pickle 回退）。
"""

from __future__ import annotations

from functools import partial

import msgpack
import numpy as np


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if isinstance(obj, dict):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = partial(msgpack.Packer, default=pack_array)
packb = partial(msgpack.packb, default=pack_array)

Unpacker = partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = partial(msgpack.unpackb, object_hook=unpack_array)
