#!/usr/bin/env python3
"""
Torch-free variant of test_engine_onesided_ipc.py for Moore Threads MUSA GPUs.

Exercises the same one-sided IPC surface (write_ipc / read_ipc, vectorized
writev_ipc / readv_ipc, and their _async + poll_async counterparts) between
two real MUSA GPUs (GPU 0 and GPU 1) on this node, using musaMalloc/musaMemcpy
(ctypes against libmusart.so) instead of torch tensors since torch_musa is
not installed in this environment.

Run with:
    python3 tests/test_engine_onesided_ipc_musa.py
"""

import ctypes
import multiprocessing
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import p2p

MUSA_HOME = os.environ.get(
    "MUSA_HOME", "/usr/local/musa-3.1.0.bak.2025-11-11-092338")
MUSART_LIB = os.path.join(MUSA_HOME, "lib", "libmusart.so")
MUSA_SUCCESS = 0
NUM_IOVS = 4
BUF_ELEMS = 1024
SIZE_PER = BUF_ELEMS * 4


def load_musart():
    return ctypes.CDLL(MUSART_LIB)


def check(err, name):
    if err != MUSA_SUCCESS:
        raise RuntimeError(f"{name} returned error code {err}")


def musa_buf(musa, gpu_idx: int, fill_val: float):
    check(musa.musaSetDevice(gpu_idx), "musaSetDevice")
    dev_ptr = ctypes.c_void_p(0)
    check(musa.musaMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(SIZE_PER)),
          "musaMalloc")
    host = (ctypes.c_float * BUF_ELEMS)(*([fill_val] * BUF_ELEMS))
    check(musa.musaMemcpy(dev_ptr, host, ctypes.c_size_t(SIZE_PER),
                           ctypes.c_int(1)), "musaMemcpy H->D")
    return dev_ptr


def musa_read(musa, gpu_idx: int, dev_ptr) -> list:
    check(musa.musaSetDevice(gpu_idx), "musaSetDevice")
    check(musa.musaDeviceSynchronize(), "musaDeviceSynchronize")
    host = (ctypes.c_float * BUF_ELEMS)()
    check(musa.musaMemcpy(host, dev_ptr, ctypes.c_size_t(SIZE_PER),
                           ctypes.c_int(2)), "musaMemcpy D->H")
    return list(host)


def allclose(vals, expected, tol=1e-5):
    return all(abs(v - expected) < tol for v in vals)


def poll_done(ep, transfer_id):
    is_done = False
    while not is_done:
        ok, is_done = ep.poll_async(transfer_id)
        assert ok, "poll_async failed"


PASS = []


def record(name, ok):
    PASS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


# ── server (rank 0, GPU 0) ───────────────────────────────────────────────────

def server_proc(pipe):
    musa = load_musart()
    ep = p2p.Endpoint(local_gpu_idx=0)
    ok, remote_gpu_idx, conn_id = ep.accept_local()
    assert ok, "accept_local failed"

    # write_ipc: client writes 1.0 into our zeroed GPU buffer.
    dst = musa_buf(musa, 0, 0.0)
    ok, info = ep.advertise_ipc(conn_id, dst.value, SIZE_PER)
    assert ok
    pipe.send(("blob", bytes(info)))
    assert pipe.recv() == "done"
    record("write_ipc", allclose(musa_read(musa, 0, dst), 1.0))

    # write_ipc_async
    dst = musa_buf(musa, 0, 0.0)
    ok, info = ep.advertise_ipc(conn_id, dst.value, SIZE_PER)
    assert ok
    pipe.send(("blob", bytes(info)))
    assert pipe.recv() == "done"
    record("write_ipc_async", allclose(musa_read(musa, 0, dst), 1.0))

    # read_ipc: server source filled with 1.0, client reads it.
    src = musa_buf(musa, 0, 1.0)
    ok, info = ep.advertise_ipc(conn_id, src.value, SIZE_PER)
    assert ok
    pipe.send(("blob", bytes(info)))
    assert pipe.recv() == "done"
    record("read_ipc (server side OK)", True)

    # read_ipc_async
    src = musa_buf(musa, 0, 1.0)
    ok, info = ep.advertise_ipc(conn_id, src.value, SIZE_PER)
    assert ok
    pipe.send(("blob", bytes(info)))
    assert pipe.recv() == "done"
    record("read_ipc_async (server side OK)", True)

    # writev_ipc: client writes [1,2,3,4] into our zeroed buffers.
    dsts = [musa_buf(musa, 0, 0.0) for _ in range(NUM_IOVS)]
    ok, infos = ep.advertisev_ipc(conn_id, [d.value for d in dsts],
                                   [SIZE_PER] * NUM_IOVS)
    assert ok
    packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in infos)
    pipe.send(("blob", packed))
    assert pipe.recv() == "done"
    ok_all = all(allclose(musa_read(musa, 0, d), float(i + 1))
                 for i, d in enumerate(dsts))
    record("writev_ipc", ok_all)

    # writev_ipc_async
    dsts = [musa_buf(musa, 0, 0.0) for _ in range(NUM_IOVS)]
    ok, infos = ep.advertisev_ipc(conn_id, [d.value for d in dsts],
                                   [SIZE_PER] * NUM_IOVS)
    assert ok
    packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in infos)
    pipe.send(("blob", packed))
    assert pipe.recv() == "done"
    ok_all = all(allclose(musa_read(musa, 0, d), float(i + 1))
                 for i, d in enumerate(dsts))
    record("writev_ipc_async", ok_all)

    # readv_ipc: server sources filled [1,2,3,4], client reads.
    srcs = [musa_buf(musa, 0, float(i + 1)) for i in range(NUM_IOVS)]
    ok, infos = ep.advertisev_ipc(conn_id, [s.value for s in srcs],
                                   [SIZE_PER] * NUM_IOVS)
    assert ok
    packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in infos)
    pipe.send(("blob", packed))
    assert pipe.recv() == "done"
    record("readv_ipc (server side OK)", True)

    # readv_ipc_async
    srcs = [musa_buf(musa, 0, float(i + 1)) for i in range(NUM_IOVS)]
    ok, infos = ep.advertisev_ipc(conn_id, [s.value for s in srcs],
                                   [SIZE_PER] * NUM_IOVS)
    assert ok
    packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in infos)
    pipe.send(("blob", packed))
    assert pipe.recv() == "done"
    record("readv_ipc_async (server side OK)", True)

    pipe.send(("results", PASS))


# ── client (rank 1, GPU 1) ───────────────────────────────────────────────────

def client_proc(pipe):
    musa = load_musart()
    ep = p2p.Endpoint(local_gpu_idx=1)
    ok, conn_id = ep.connect_local(remote_gpu_bdf=os.environ["SERVER_GPU_BDF"])
    assert ok, "connect_local failed"

    # write_ipc
    kind, info = pipe.recv()
    src = musa_buf(musa, 1, 1.0)
    ok = ep.write_ipc(conn_id, src.value, SIZE_PER, info)
    assert ok, "write_ipc failed"
    pipe.send("done")

    # write_ipc_async
    kind, info = pipe.recv()
    src = musa_buf(musa, 1, 1.0)
    ok, tid = ep.write_ipc_async(conn_id, src.value, SIZE_PER, info)
    assert ok, "write_ipc_async failed"
    poll_done(ep, tid)
    pipe.send("done")

    # read_ipc
    kind, info = pipe.recv()
    dst = musa_buf(musa, 1, 0.0)
    ok = ep.read_ipc(conn_id, dst.value, SIZE_PER, info)
    assert ok, "read_ipc failed"
    record("read_ipc", allclose(musa_read(musa, 1, dst), 1.0))
    pipe.send("done")

    # read_ipc_async
    kind, info = pipe.recv()
    dst = musa_buf(musa, 1, 0.0)
    ok, tid = ep.read_ipc_async(conn_id, dst.value, SIZE_PER, info)
    assert ok, "read_ipc_async failed"
    poll_done(ep, tid)
    record("read_ipc_async", allclose(musa_read(musa, 1, dst), 1.0))
    pipe.send("done")

    # writev_ipc
    kind, packed = pipe.recv()
    n = struct.unpack_from("I", packed, 0)[0]
    blob_size = (len(packed) - 4) // n
    infos = [packed[4 + i * blob_size: 4 + (i + 1) * blob_size] for i in range(n)]
    srcs = [musa_buf(musa, 1, float(i + 1)) for i in range(n)]
    ok = ep.writev_ipc(conn_id, [s.value for s in srcs], [SIZE_PER] * n, infos)
    assert ok, "writev_ipc failed"
    pipe.send("done")

    # writev_ipc_async
    kind, packed = pipe.recv()
    n = struct.unpack_from("I", packed, 0)[0]
    blob_size = (len(packed) - 4) // n
    infos = [packed[4 + i * blob_size: 4 + (i + 1) * blob_size] for i in range(n)]
    srcs = [musa_buf(musa, 1, float(i + 1)) for i in range(n)]
    ok, tid = ep.writev_ipc_async(conn_id, [s.value for s in srcs],
                                   [SIZE_PER] * n, infos)
    assert ok, "writev_ipc_async failed"
    poll_done(ep, tid)
    pipe.send("done")

    # readv_ipc
    kind, packed = pipe.recv()
    n = struct.unpack_from("I", packed, 0)[0]
    blob_size = (len(packed) - 4) // n
    infos = [packed[4 + i * blob_size: 4 + (i + 1) * blob_size] for i in range(n)]
    dsts = [musa_buf(musa, 1, 0.0) for _ in range(n)]
    ok = ep.readv_ipc(conn_id, [d.value for d in dsts], [SIZE_PER] * n, infos)
    assert ok, "readv_ipc failed"
    ok_all = all(allclose(musa_read(musa, 1, d), float(i + 1))
                 for i, d in enumerate(dsts))
    record("readv_ipc", ok_all)
    pipe.send("done")

    # readv_ipc_async
    kind, packed = pipe.recv()
    n = struct.unpack_from("I", packed, 0)[0]
    blob_size = (len(packed) - 4) // n
    infos = [packed[4 + i * blob_size: 4 + (i + 1) * blob_size] for i in range(n)]
    dsts = [musa_buf(musa, 1, 0.0) for _ in range(n)]
    ok, tid = ep.readv_ipc_async(conn_id, [d.value for d in dsts],
                                  [SIZE_PER] * n, infos)
    assert ok, "readv_ipc_async failed"
    poll_done(ep, tid)
    ok_all = all(allclose(musa_read(musa, 1, d), float(i + 1))
                 for i, d in enumerate(dsts))
    record("readv_ipc_async", ok_all)
    pipe.send("done")

    pipe.send(("results", PASS))


def main():
    multiprocessing.set_start_method("spawn", force=True)

    musa = load_musart()
    bus_id = ctypes.create_string_buffer(64)
    check(musa.musaDeviceGetPCIBusId(bus_id, ctypes.c_int(64), ctypes.c_int(0)),
          "musaDeviceGetPCIBusId")
    os.environ["SERVER_GPU_BDF"] = bus_id.value.decode().lower()
    print(f"Server GPU 0 bus id: {os.environ['SERVER_GPU_BDF']}")

    srv_parent, srv_child = multiprocessing.Pipe()
    cli_parent, cli_child = multiprocessing.Pipe()

    srv = multiprocessing.Process(target=server_proc, args=(srv_child,))
    cli = multiprocessing.Process(target=client_proc, args=(cli_child,))
    srv.start()
    time.sleep(1)
    cli.start()

    # Relay messages between the two pipes (server <-> client) in this
    # parent process since accept_local/connect_local talk over a local
    # control channel, not directly via these pipes.
    pending_blob = None
    results = {"server": None, "client": None}
    done_count = 0
    while True:
        if srv_parent.poll(0.01):
            msg = srv_parent.recv()
            if msg[0] == "blob":
                cli_parent.send(("blob", msg[1]))
            elif msg[0] == "results":
                results["server"] = msg[1]
        if cli_parent.poll(0.01):
            msg = cli_parent.recv()
            if msg == "done":
                srv_parent.send("done")
            elif isinstance(msg, tuple) and msg[0] == "results":
                results["client"] = msg[1]
        if results["server"] is not None and results["client"] is not None:
            break
        if not srv.is_alive() and not cli.is_alive():
            break

    srv.join(timeout=10)
    cli.join(timeout=10)

    all_results = (results["server"] or []) + (results["client"] or [])
    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    for name, ok in all_results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed = [n for n, ok in all_results if not ok]
    assert srv.exitcode == 0, f"server exitcode={srv.exitcode}"
    assert cli.exitcode == 0, f"client exitcode={cli.exitcode}"
    assert not failed, f"failed tests: {failed}"
    assert len(all_results) >= 12, f"too few results: {len(all_results)}"
    print("\nAll one-sided IPC tests passed! (MUSA, torch-free)")


if __name__ == "__main__":
    main()
