#!/usr/bin/env python3
"""
Basic Moore Threads MUSA P2P validation — no torch required.

Tests via ctypes directly against libmusart.so/libmusa.so:
  1. GPU enumeration
  2. musaDeviceCanAccessPeer between GPU 0 and GPU 1
  3. musaIpcGetMemHandle + musaIpcOpenMemHandle (the P2P IPC path)
  4. musaMemcpyPeer between GPU 0 and GPU 1
  5. p2p module load + Endpoint creation
"""

import ctypes
import multiprocessing
import os
import sys

MUSA_HOME = os.environ.get(
    "MUSA_HOME", "/usr/local/musa-3.1.0.bak.2025-11-11-092338")
MUSART_LIB = os.path.join(MUSA_HOME, "lib", "libmusart.so")
MUSA_SUCCESS = 0


def load_musart():
    try:
        lib = ctypes.CDLL(MUSART_LIB)
    except OSError as e:
        sys.exit(f"Cannot load {MUSART_LIB}: {e}")
    return lib


def check(err, name):
    if err != MUSA_SUCCESS:
        sys.exit(f"[FAIL] {name} returned error code {err}")


# ── 1. GPU enumeration ───────────────────────────────────────────────────────

def test_device_count(musa):
    count = ctypes.c_int(0)
    check(musa.musaGetDeviceCount(ctypes.byref(count)), "musaGetDeviceCount")
    n = count.value
    print(f"[PASS] musaGetDeviceCount -> {n} GPU(s)")
    if n < 1:
        sys.exit("No Moore Threads GPUs found")
    return n


# ── 2. Peer access capability ────────────────────────────────────────────────

def test_peer_access(musa, n):
    if n < 2:
        print("[SKIP] peer access check - only 1 GPU")
        return
    can = ctypes.c_int(0)
    check(musa.musaDeviceCanAccessPeer(ctypes.byref(can), 0, 1),
          "musaDeviceCanAccessPeer(0->1)")
    print(f"[{'PASS' if can.value else 'WARN'}] "
          f"musaDeviceCanAccessPeer(0->1) = {bool(can.value)}")


# ── 3. IPC memory handle round-trip ──────────────────────────────────────────

IPC_HANDLE_SIZE = 64  # sizeof(musaIpcMemHandle_t)


class musaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_uint8 * IPC_HANDLE_SIZE)]


def _ipc_child(child_conn):
    """Open the IPC handle sent from the parent on GPU 1 and verify data."""
    musa = load_musart()
    check(musa.musaSetDevice(1), "musaSetDevice(1) [child]")

    raw = child_conn.recv()  # bytes: handle (64 bytes)
    handle = musaIpcMemHandle_t()
    ctypes.memmove(ctypes.addressof(handle), raw, IPC_HANDLE_SIZE)

    ptr = ctypes.c_void_p(0)
    err = musa.musaIpcOpenMemHandle(ctypes.byref(ptr), handle,
                                     ctypes.c_uint(0))
    if err != MUSA_SUCCESS:
        child_conn.send(f"FAIL:musaIpcOpenMemHandle err={err}")
        return

    HOST_BYTES = 4 * 4  # 4 floats
    host_buf = (ctypes.c_float * 4)()
    err = musa.musaMemcpy(host_buf, ptr, ctypes.c_size_t(HOST_BYTES),
                           ctypes.c_int(2))  # musaMemcpyDeviceToHost
    if err != MUSA_SUCCESS:
        child_conn.send(f"FAIL:musaMemcpy err={err}")
        return

    values = list(host_buf)
    musa.musaIpcCloseMemHandle(ptr)
    child_conn.send(f"OK:{values}")


def test_ipc_handles(musa, n):
    if n < 2:
        print("[SKIP] IPC handle test - only 1 GPU")
        return

    check(musa.musaSetDevice(0), "musaSetDevice(0)")

    HOST_BYTES = 4 * 4
    src_host = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
    dev_ptr = ctypes.c_void_p(0)
    check(musa.musaMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(HOST_BYTES)),
          "musaMalloc")
    check(musa.musaMemcpy(dev_ptr, src_host, ctypes.c_size_t(HOST_BYTES),
                           ctypes.c_int(1)),  # musaMemcpyHostToDevice
          "musaMemcpy H->D")

    handle = musaIpcMemHandle_t()
    check(musa.musaIpcGetMemHandle(ctypes.byref(handle), dev_ptr),
          "musaIpcGetMemHandle")
    print("[PASS] musaIpcGetMemHandle succeeded on GPU 0")

    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=_ipc_child, args=(child_conn,))
    proc.start()
    parent_conn.send(bytes(bytearray(handle.reserved)))
    result = parent_conn.recv()
    proc.join()

    if result.startswith("OK:"):
        values = eval(result[3:])
        ok = all(abs(v - 1.0) < 1e-5 for v in values)
        print(f"[{'PASS' if ok else 'FAIL'}] "
              f"musaIpcOpenMemHandle on GPU 1 -> values={values}")
    else:
        print(f"[FAIL] {result}")

    musa.musaFree(dev_ptr)


# ── 4. musaMemcpyPeer ────────────────────────────────────────────────────────

def test_memcpy_peer(musa, n):
    if n < 2:
        print("[SKIP] musaMemcpyPeer - only 1 GPU")
        return

    NBYTES = 4 * 4
    src_host = (ctypes.c_float * 4)(2.0, 2.0, 2.0, 2.0)
    dst_host = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)

    check(musa.musaSetDevice(0), "musaSetDevice(0)")
    src_dev = ctypes.c_void_p(0)
    check(musa.musaMalloc(ctypes.byref(src_dev), ctypes.c_size_t(NBYTES)),
          "musaMalloc src")
    check(musa.musaMemcpy(src_dev, src_host, ctypes.c_size_t(NBYTES),
                           ctypes.c_int(1)), "musaMemcpy H->D src")

    check(musa.musaSetDevice(1), "musaSetDevice(1)")
    dst_dev = ctypes.c_void_p(0)
    check(musa.musaMalloc(ctypes.byref(dst_dev), ctypes.c_size_t(NBYTES)),
          "musaMalloc dst")

    err = musa.musaMemcpyPeer(dst_dev, ctypes.c_int(1), src_dev,
                               ctypes.c_int(0), ctypes.c_size_t(NBYTES))
    if err != MUSA_SUCCESS:
        print(f"[WARN] musaMemcpyPeer failed (err={err}) - "
              "peer access may not be enabled")
    else:
        check(musa.musaMemcpy(dst_host, dst_dev, ctypes.c_size_t(NBYTES),
                               ctypes.c_int(2)), "musaMemcpy D->H dst")
        ok = all(abs(dst_host[i] - 2.0) < 1e-5 for i in range(4))
        print(f"[{'PASS' if ok else 'FAIL'}] "
              f"musaMemcpyPeer GPU0->GPU1 -> {list(dst_host)}")

    musa.musaFree(src_dev)
    musa.musaFree(dst_dev)


# ── 5. p2p module load ───────────────────────────────────────────────────────

def test_p2p_module():
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import p2p
        ep = p2p.Endpoint(local_gpu_idx=0)
        print(f"[PASS] p2p module loaded and Endpoint created: {ep}")
    except Exception as e:
        print(f"[FAIL] p2p module/Endpoint creation failed: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    print("=" * 60)
    print("  Moore Threads MUSA P2P Basic Test")
    print("=" * 60)

    musa = load_musart()
    n = test_device_count(musa)
    test_peer_access(musa, n)
    test_ipc_handles(musa, n)
    test_memcpy_peer(musa, n)
    test_p2p_module()

    print("=" * 60)
    print("  Done")
    print("=" * 60)
