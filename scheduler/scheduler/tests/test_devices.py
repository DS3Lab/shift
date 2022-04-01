from typing import Sequence

import pytest

from .._devices import DeviceManager


class TestDeviceManager:
    def test_gpu_ids_from_string(self):
        def assert_equal_result(string: str, expected: Sequence[str]):
            assert set(DeviceManager.get_gpu_ids_from_string(string)) == set(expected)

        assert_equal_result("", [])
        assert_equal_result("0", ["0"])
        assert_equal_result(",,0", ["0"])
        assert_equal_result(",1,1,,", ["0"])
        assert_equal_result("0,   ,1 , 2  ,2 , 4", ["0", "1", "2", "3"])

    def test_release_device(self):
        dm = DeviceManager(gpu_ids_string="0,1,2", max_cpu_jobs=1)

        # 1. OK to release GPU that was free
        dm.release_device("0")

        # 2. Not OK to release CPU when no was reserved
        with pytest.raises(RuntimeError):
            dm.release_device(None)

        # 3. Not OK to release inexistent GPU
        with pytest.raises(ValueError):
            dm.release_device("10")

        # 4. Regular operation
        gpu_id = dm.get_free_gpu()
        cpu_id = dm.get_free_cpu()

        dm.release_device(gpu_id)
        dm.release_device(cpu_id)

    def test_no_gpus_available(self):
        dm = DeviceManager(gpu_ids_string="", max_cpu_jobs=2)

        # No GPUs -> CPUs presented as GPUs
        # 1. Initial state
        assert dm.any_gpu_free()
        assert dm.any_cpu_free()

        # 2. Reserve all CPUs
        assert dm.get_free_gpu() is None
        assert dm.get_free_cpu() is None

        assert not dm.any_gpu_free()
        assert not dm.any_cpu_free()

        # 3. Reserve more than available
        with pytest.raises(RuntimeError):
            dm.get_free_cpu()
        with pytest.raises(RuntimeError):
            dm.get_free_gpu()

    def test_gpus_available(self):
        dm = DeviceManager(gpu_ids_string="0,1", max_cpu_jobs=0)

        # 1. Initial state
        assert dm.any_gpu_free()
        assert not dm.any_cpu_free()

        # 2. Reserve all GPUs
        assert dm.get_free_gpu() in {"0", "1"}
        assert dm.get_free_gpu() in {"0", "1"}

        # 3. Reserve more than available
        with pytest.raises(RuntimeError):
            dm.get_free_cpu()
        with pytest.raises(RuntimeError):
            dm.get_free_gpu()
