#!/usr/bin/env python3
"""Cross-node GPU-memory RDMA-WRITE test for Moore Threads MUSA GPUs.

Run this on TWO machines that share an RDMA-capable network. One side is the
receiver (accepts the connection and exposes a GPU buffer for one-sided
RDMA-WRITE); the other side is the sender (connects and writes 1.0 into the
receiver's buffer). A small out-of-band TCP channel exchanges the UCCL
endpoint metadata and the one-sided FIFO blob.

Receiver (node A):
    python3 test_engine_write_musa_xnode.py --role recv \
        --bind 0.0.0.0 --oob-port 18888 --gpu 0

Sender (node B), pointing at node A's OOB address:
    python3 test_engine_write_musa_xnode.py --role send \
        --peer <NODE_A_IP> --oob-port 18888 --gpu 0

The receiver verifies its GPU buffer changed 0.0 -> 1.0; the sender verifies
its source buffer stayed 1.0. Exit code 0 == success on that node.

Torch-free: GPU buffers are allocated via musaMalloc/musaMemcpy (ctypes
against libmusart.so) since this environment has no torch_musa.
"""
from __future__ import annotations
import argparse
import ctypes
import os
import socket
import struct
import sys
import time

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


def musa_alloc_filled(musa, value: float, gpu_idx: int):
    check(musa.musaSetDevice(gpu_idx), "musaSetDevice")
    dev_ptr = ctypes.c_void_p(0)
    check(musa.musaMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(N_BYTES)),
          "musaMalloc")
    host_buf = (ctypes.c_float * N_FLOATS)(*([value] * N_FLOATS))
    check(musa.musaMemcpy(dev_ptr, host_buf, ctypes.c_size_t(N_BYTES),
                          ctypes.c_int(1)), "musaMemcpy H->D")  # H2D
    return dev_ptr


def musa_read_back(musa, dev_ptr) -> list:
    check(musa.musaDeviceSynchronize(), "musaDeviceSynchronize")
    host_buf = (ctypes.c_float * N_FLOATS)()
    check(musa.musaMemcpy(host_buf, dev_ptr, ctypes.c_size_t(N_BYTES),
                          ctypes.c_int(2)), "musaMemcpy D->H")  # D2H
    return list(host_buf)


# ── out-of-band length-prefixed messaging ────────────────────────────────────

def send_msg(sock, payload: bytes):
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_n(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("OOB channel closed early")
        buf += chunk
    return buf


def recv_msg(sock) -> bytes:
    (length,) = struct.unpack("!I", recv_n(sock, 4))
    return recv_n(sock, length)


# ── receiver: accept + expose buffer for one-sided RDMA-WRITE ─────────────────

def run_recv(args):
    musa = load_musart()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.oob_port))
    srv.listen(1)
    print(f"[recv] OOB listening on {args.bind}:{args.oob_port} ...")
    conn, peer = srv.accept()
    print(f"[recv] OOB connected from {peer}")

    ep = p2p.Endpoint(local_gpu_idx=args.gpu)
    send_msg(conn, bytes(ep.get_metadata()))
    print("[recv] sent endpoint metadata, waiting for UCCL accept ...")

    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "accept failed"
    print(f"[recv] UCCL connected (from {r_ip}, conn_id={conn_id})")

    dev_ptr = musa_alloc_filled(musa, 0.0, args.gpu)
    ok, mr_id = ep.reg(dev_ptr.value, N_BYTES)
    assert ok, "reg failed"

    ok, fifo_blob = ep.advertise(mr_id, dev_ptr.value, N_BYTES)
    assert isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob) == 64
    send_msg(conn, bytes(fifo_blob))
    print("[recv] buffer exposed, FIFO blob sent; waiting for sender 'done' ...")

    assert recv_msg(conn) == b"done", "sender did not report done"

    values = musa_read_back(musa, dev_ptr)
    print(f"[recv] buffer[:8] after RDMA-WRITE: {values[:8]}")
    ok = all(abs(v - 1.0) < 1e-5 for v in values)
    send_msg(conn, b"ok" if ok else b"bad")
    conn.close()
    srv.close()
    if not ok:
        raise AssertionError("receiver buffer was NOT overwritten by RDMA-WRITE")
    print("[recv] PASS: cross-node GPU RDMA-WRITE received correctly")


# ── sender: connect + RDMA-WRITE into receiver's advertised buffer ────────────

def run_send(args):
    musa = load_musart()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[send] OOB connecting to {args.peer}:{args.oob_port} ...")
    for attempt in range(30):
        try:
            sock.connect((args.peer, args.oob_port))
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    else:
        raise ConnectionError("could not reach receiver OOB port")
    print("[send] OOB connected")

    ep_meta = recv_msg(sock)
    ip, port, r_gpu = p2p.Endpoint.parse_metadata(ep_meta)
    print(f"[send] receiver metadata: ip={ip} port={port} gpu={r_gpu}")

    ep = p2p.Endpoint(local_gpu_idx=args.gpu)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "connect failed"
    print(f"[send] UCCL connected (conn_id={conn_id})")

    dev_ptr = musa_alloc_filled(musa, 1.0, args.gpu)
    ok, mr_id = ep.reg(dev_ptr.value, N_BYTES)
    assert ok, "reg failed"

    fifo_blob = recv_msg(sock)
    assert len(fifo_blob) == 64, "bad FIFO blob"
    ok = ep.write(conn_id, mr_id, dev_ptr.value, N_BYTES, fifo_blob)
    assert ok, "write failed"
    print("[send] RDMA-WRITE issued")

    send_msg(sock, b"done")
    verdict = recv_msg(sock)
    sock.close()

    values = musa_read_back(musa, dev_ptr)
    assert all(abs(v - 1.0) < 1e-5 for v in values), "source buffer corrupted"
    if verdict != b"ok":
        raise AssertionError("receiver reported data mismatch")
    print("[send] PASS: receiver confirmed correct RDMA-WRITE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["recv", "send"])
    ap.add_argument("--peer", help="receiver IP (sender only)")
    ap.add_argument("--bind", default="0.0.0.0", help="OOB bind addr (recv)")
    ap.add_argument("--oob-port", type=int, default=18888)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.role == "recv":
        run_recv(args)
    else:
        if not args.peer:
            ap.error("--peer is required for --role send")
        run_send(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
