"""Making physical room on the card.

The OS can page another process's VRAM out to system RAM, but nothing
demotes memory on request: only allocation pressure triggers it. So the way
to free the card is to claim the bytes and hand them straight back. The
driver has to find physical memory for the claim, demoting other processes'
allocations to do it, and releasing the claim leaves that memory free for
the engine that starts next. The demoted processes keep working from system
RAM until they touch the card again; they lose performance, never
correctness.

This matters for engines that size themselves from a measurement before
allocating anything: they see the occupied card, conclude the model does not
fit, and give up without ever applying the pressure that would have made it
fit.

``RoomMaker`` is the abstraction the orchestrator depends on.
``CudaRoomMaker`` implements it against the CUDA driver API through ctypes,
which costs no dependency and works wherever the driver is installed. The
claim is chunked: a partial claim still frees partial room, and the caller
re-measures rather than trusting the return value.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Protocol


class RoomMaker(Protocol):
    def make_room(self, target_bytes: int, gpu_index: int = 0) -> int: ...


class CudaRoomMaker:  # pragma: no cover - needs a GPU and the CUDA driver
    _CHUNK_BYTES = 256 * 2**20

    def make_room(self, target_bytes: int, gpu_index: int = 0) -> int:
        """Claim ``target_bytes`` of VRAM and release them; returns bytes claimed."""
        cuda = self._driver()
        self._check("cuInit", cuda.cuInit(0))
        device = ctypes.c_int()
        self._check("cuDeviceGet", cuda.cuDeviceGet(ctypes.byref(device), gpu_index))
        context = ctypes.c_void_p()
        self._check(
            "cuDevicePrimaryCtxRetain",
            cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
        )
        try:
            self._check("cuCtxPushCurrent", cuda.cuCtxPushCurrent_v2(context))
            try:
                return self._claim_and_release(cuda, target_bytes)
            finally:
                popped = ctypes.c_void_p()
                cuda.cuCtxPopCurrent_v2(ctypes.byref(popped))
        finally:
            cuda.cuDevicePrimaryCtxRelease_v2(device)

    def _claim_and_release(self, cuda, target_bytes: int) -> int:
        pointers: list[ctypes.c_uint64] = []
        claimed = 0
        try:
            while claimed < target_bytes:
                chunk = min(self._CHUNK_BYTES, target_bytes - claimed)
                pointer = ctypes.c_uint64()
                # A refused chunk is not an error: the card gave what it
                # could, partial room is still room, and the caller measures.
                if cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(chunk)):
                    break
                pointers.append(pointer)
                claimed += chunk
        finally:
            for pointer in pointers:
                cuda.cuMemFree_v2(pointer)
        return claimed

    @staticmethod
    def _driver():
        if sys.platform == "win32":
            return ctypes.WinDLL("nvcuda.dll")
        return ctypes.CDLL("libcuda.so.1")

    @staticmethod
    def _check(call: str, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"CUDA driver call {call} failed with error {code}")
