#!/usr/bin/env python3
"""
Local unit-test for UCCL P2P Engine on Moore Threads MUSA GPUs — server
writes data with RDMA-WRITE using the one-sided metadata handshake.

Torch-free variant of test_engine_write.py: device buffers are allocated
directly via musaMalloc/musaMemcpy (ctypes against libmusart.so) instead of
torch tensors, since this environment does not have torch_musa installed.
"""

from __future__ import annotations
import ctypes
import multiprocessing
import os
import sys
import time
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    import p2p
except ImportError as e:
    sys.stderr.write(f"Failed to import p2p: {e}\n")
    raise

MUSA_HOME = os.environ.get(
    "MUSA_HOME", "/usr/local/musa-3.1.0.bak.2025-11-11-092338")
MUSART_LIB = os.path.join(MUSA_HOME, "lib", "libmusart.so")
MUSA_SUCCESS = 0
N_FLOATS = 1024
N_BYTES = N_FLOATS * 4


def load_musart():
    return ctypes.CDLL(MUSART_LIB)


def check(err, name):
    if err != MUSA_SUCCESS:
        raise RuntimeError(f"{name} returned error code {err}")


def musa_alloc_filled(musa, value: float, gpu_idx: int = 0):
    """musaMalloc N_BYTES on gpu_idx, fill with `value`, return device ptr."""
    check(musa.musaSetDevice(gpu_idx), "musaSetDevice")
    dev_ptr = ctypes.c_void_p(0)
    check(musa.musaMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(N_BYTES)),
          "musaMalloc")
    host_buf = (ctypes.c_float * N_FLOATS)(*([value] * N_FLOATS))
    check(musa.musaMemcpy(dev_ptr, host_buf, ctypes.c_size_t(N_BYTES),
                           ctypes.c_int(1)),  # musaMemcpyHostToDevice
          "musaMemcpy H->D")
    return dev_ptr


def musa_read_back(musa, dev_ptr) -> list:
    check(musa.musaDeviceSynchronize(), "musaDeviceSynchronize")
    host_buf = (ctypes.c_float * N_FLOATS)()
    check(musa.musaMemcpy(host_buf, dev_ptr, ctypes.c_size_t(N_BYTES),
                           ctypes.c_int(2)),  # musaMemcpyDeviceToHost
          "musaMemcpy D->H")
    return list(host_buf)


def parse_endpoint_meta(meta: bytes) -> Tuple[str, int, str]:
    return p2p.Endpoint.parse_metadata(meta)


def test_local():
    print("Running RDMA-WRITE local test (MUSA, torch-free)")
    meta_parent, meta_child = multiprocessing.Pipe()
    fifo_parent, fifo_child = multiprocessing.Pipe()

    def server_proc(ep_meta_q, fifo_meta_q):
        musa = load_musart()
        ep_meta = ep_meta_q.recv()
        ip, port, r_gpu = parse_endpoint_meta(ep_meta)

        ep = p2p.Endpoint(local_gpu_idx=0)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
        assert ok, "connect failed"
        print(f"[Server] connected (conn_id={conn_id})")

        # Server's buffer starts at 1.0 — this is the data RDMA-WRITE will
        # push into the client's advertised buffer.
        dev_ptr = musa_alloc_filled(musa, 1.0, gpu_idx=0)
        ok, mr_id = ep.reg(dev_ptr.value, N_BYTES)
        assert ok

        fifo_meta = fifo_meta_q.recv()
        assert isinstance(fifo_meta, (bytes, bytearray)) and len(fifo_meta) == 64

        ok = ep.write(conn_id, mr_id, dev_ptr.value, N_BYTES, fifo_meta)
        assert ok, "write failed"

        values = musa_read_back(musa, dev_ptr)
        print("server tensor[:8]:", values[:8])
        assert all(abs(v - 1.0) < 1e-5 for v in values)
        print("Server write data correctly")

    def client_proc(ep_meta_q, fifo_meta_q):
        musa = load_musart()
        ep = p2p.Endpoint(local_gpu_idx=0)
        ep_meta_q.send(bytes(ep.get_metadata()))

        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "accept failed"
        print(f"[Client] accepted (conn_id={conn_id})")

        # Client's buffer starts at 0.0 — after RDMA-WRITE from the server
        # it should read back as 1.0.
        dev_ptr = musa_alloc_filled(musa, 0.0, gpu_idx=0)

        ok, mr_id = ep.reg(dev_ptr.value, N_BYTES)
        assert ok
        time.sleep(0.1)
        ok, fifo_blob = ep.advertise(mr_id, dev_ptr.value, N_BYTES)
        assert isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob) == 64
        print("Buffer exposed for RDMA WRITE")

        fifo_meta_q.send(bytes(fifo_blob))
        time.sleep(1)

        values = musa_read_back(musa, dev_ptr)
        print("client buffer[:8] after RDMA-WRITE:", values[:8])
        assert all(abs(v - 1.0) < 1e-5 for v in values), (
            "client buffer was not overwritten by RDMA-WRITE")
        print("Client buffer received RDMA-WRITE correctly")

    srv = multiprocessing.Process(target=server_proc,
                                   args=(meta_parent, fifo_parent))
    cli = multiprocessing.Process(target=client_proc,
                                   args=(meta_child, fifo_child))
    srv.start()
    time.sleep(1)
    cli.start()
    srv.join()
    cli.join()
    assert srv.exitcode == 0, f"server failed (exitcode={srv.exitcode})"
    assert cli.exitcode == 0, f"client failed (exitcode={cli.exitcode})"
    print("Local RDMA-WRITE test passed (MUSA)\n")


if __name__ == "__main__":
    try:
        test_local()
    except KeyboardInterrupt:
        print("\nInterrupted, terminating...")
        sys.exit(1)
