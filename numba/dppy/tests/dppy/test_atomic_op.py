from __future__ import print_function, division, absolute_import

import numpy as np

import numba
from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import dppy.core as ocldrv


def atomic_add(ary):
    tid = dppy.get_local_id(0)
    lm = dppy.local.alloc(32, numba.uint32)
    lm[tid] = 0
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    bin = ary[tid] % 32
    dppy.atomic.add(lm, bin, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[tid] = lm[tid]


def atomic_add2(ary):
    tx = dppy.get_local_id(0)
    ty = dppy.get_local_id(1)
    lm = dppy.local.alloc((4, 8), numba.uint32)
    lm[tx, ty] = ary[tx, ty]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.add(lm, (tx, ty), 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = lm[tx, ty]


def atomic_add3(ary):
    tx = dppy.get_local_id(0)
    ty = dppy.get_local_id(1)
    lm = dppy.local.alloc((4, 8), numba.uint32)
    lm[tx, ty] = ary[tx, ty]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.add(lm, (tx, numba.uint64(ty)), 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = lm[tx, ty]



class TestAtomicOp(DPPYTestCase):
    gpu_device_env = None

    try:
        gpu_device_env = ocldrv.runtime.get_gpu_device()
    except:
        print("GPU device not found")

    def test_atomic_add(self):
        @dppy.kernel
        def atomic_add(B):
            i = dppy.get_global_id(0)
            dppy.atomic.add(B, 0, 1)

        N = 100
        B = np.array([0])

        atomic_add[self.gpu_device_env, N](B)

        self.assertTrue(B[0] == N)


    def test_atomic_sub(self):
        @dppy.kernel
        def atomic_sub(B):
            i = dppy.get_global_id(0)
            dppy.atomic.sub(B, 0, 1)

        N = 100
        B = np.array([100])

        atomic_sub[self.gpu_device_env, N](B)

        self.assertTrue(B[0] == 0)

    def test_atomic_add1(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()
        dppy_atomic_add = dppy.kernel('void(uint32[:])')(atomic_add)
        dppy_atomic_add[self.gpu_device_env, 32](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        dppy_atomic_add2 = dppy.kernel('void(uint32[:,:])')(atomic_add2)
        dppy_atomic_add2[self.gpu_device_env, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        dppy_atomic_add3 = dppy.kernel('void(uint32[:,:])')(atomic_add3)
        dppy_atomic_add3[self.gpu_device_env, (4, 8)](ary)

        self.assertTrue(np.all(ary == orig + 1))


if __name__ == '__main__':
    unittest.main()
