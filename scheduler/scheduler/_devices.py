import logging
from typing import Optional, Sequence

DeviceID = Optional[str]

_logger = logging.getLogger(__name__)


class DeviceManager:
    """Tracks which CPU and GPU devices are currently in use and which are free. If no GPU device is specified, GPU jobs are treated as CPU jobs, meaning that the results will give an indication that there are free GPUs even though there are not GPUs available.

    Args:
        gpu_ids_string (str): Specification of GPU devices to use.
        max_cpu_jobs (int): Maximal number of concurrent CPU jobs.
    """

    def __init__(self, gpu_ids_string: str, max_cpu_jobs: int):
        gpu_ids = self.get_gpu_ids_from_string(gpu_ids_string)
        _logger.info(
            "Using GPU devices %r and allowing %s cpu jobs", gpu_ids, max_cpu_jobs
        )
        if len(gpu_ids) > 0:
            self._is_gpu_free = {gpu_id: True for gpu_id in gpu_ids}
            self._cpu_only = False
        else:
            self._is_gpu_free = {}
            self._cpu_only = True
        self._cpu_jobs = 0
        self._max_cpu_jobs = max_cpu_jobs

    @property
    def _cpu_free(self) -> bool:
        return self._cpu_jobs < self._max_cpu_jobs

    def any_gpu_free(self) -> bool:
        """True if any GPU device is free, False otherwise. In case that there are
        no GPU devices specified, the result indicates whether there is a free CPU
        slot."""
        return self._cpu_free if self._cpu_only else True in self._is_gpu_free.values()

    def any_cpu_free(self) -> bool:
        """True if any CPU device is free, False otherwise."""
        return self._cpu_free

    def get_free_gpu(self) -> DeviceID:
        """Requests a free GPU device or a CPU slot if there are no GPU devices
        specified.

        Returns:
            DeviceID: ID of the free device.
        """
        if self._cpu_only:
            if self._cpu_free:
                self._cpu_jobs += 1
                return None

        else:
            for gpu_id in self._is_gpu_free:
                if self._is_gpu_free[gpu_id]:
                    self._is_gpu_free[gpu_id] = False
                    _logger.debug("Reserved GPU device %s", gpu_id)
                    return gpu_id

        raise RuntimeError("Free GPU requested when there was no free GPU")

    def get_free_cpu(self) -> DeviceID:
        """Requests a free CPU slot.

        Returns:
            DeviceID: Value None, which denotes the use of a CPU.
        """
        if self._cpu_free:
            self._cpu_jobs += 1
            _logger.debug("Reserved CPU device")
            return None

        raise RuntimeError(
            "CPU requested, but but the maximal number of jobs is reached"
        )

    def release_device(self, device_id: DeviceID):
        """Makes the specified device available for other jobs.

        Args:
            device_id (DeviceID): Device to release.
        """
        if device_id is None:
            if self._cpu_jobs <= 0:
                raise RuntimeError("There is no CPU to release")

            self._cpu_jobs -= 1

        elif device_id in self._is_gpu_free:
            self._is_gpu_free[device_id] = True

        else:
            raise ValueError(
                f"Device string {device_id!r} unknown, cannot be released."
            )
        _logger.debug("Released device %s", device_id)

    @staticmethod
    def get_gpu_ids_from_string(gpu_ids_string: str) -> Sequence[DeviceID]:
        """Parses string which specifies which GPU devices to use. The resulting strings can be used to set environment variable 'CUDA_VISIBLE_DEVICES'.
        The returned device IDs always look like '0','1','2',... even if the specified devices have different IDs.

        This is because when using 'device_ids' property within
        docker-compose GPU devices are remapped to start with 0.

        > (xzyao) Updated: Now it runs on baremetal without docker-compose, so we don't remap the id to avoid conflicts with others

        Args:
            gpu_ids_string (str): Specification of GPU devices to use.

        Returns:
            Sequence[DeviceID]: GPU IDs that will be used to select a GPU.
        """
        # This works, because docker-compose will reject invalid IDs
        # However strings ',,', '0,,', '0,0', '0,' are permitted

        # Undefined -> CPU only
        if gpu_ids_string.strip() == "":
            return []

        # 1. Split by "," and remove unneeded spaces
        step_1 = map(lambda x: x.strip(), gpu_ids_string.split(","))

        # 2. Remove "" entries
        step_2 = filter(lambda x: len(x) > 0, step_1)

        # 3. Remove duplicates
        step_3 = list(set(step_2))

        # 4. Generate new IDs
        return [str(i) for i in step_3]

    def __str__(self):
        free_gpu_ids = [i for i in self._is_gpu_free if self._is_gpu_free[i]]
        taken_gpu_ids = [i for i in self._is_gpu_free if not self._is_gpu_free[i]]

        return (
            f"Free GPUs: {free_gpu_ids}, Taken GPUs: {taken_gpu_ids}, "
            f"Free CPU slots: {self._max_cpu_jobs - self._cpu_jobs}"
        )
