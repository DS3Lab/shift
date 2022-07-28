import multiprocessing
from collections import OrderedDict
from math import ceil, floor, log
from typing import List
import itertools
from db_tools.postgres import JobsDB
from loguru import logger

from ._base import Arm
from .observers import ShiftObserver
from .processes import ArmProcess


def get_bounds(data_size, num_models, eta=2):
    """
    budget: is determined by the num_models
    we only need to ensure, at the first step, there is enough budget. i.e., r_k>=1
    """
    # in the first step, S_ks[0] = num_models
    budget = num_models * ceil(log(num_models, eta))
    """
    chunk size is more complex: we need the total number of pulls
    """
    total_pulls = 0
    S_ks = [num_models]
    r_ks = []
    for k in range(ceil(log(num_models, eta))):
        r_k = floor(budget/(S_ks[k] * ceil(log(num_models, eta))))
        r_ks.append(r_k)
        R_k = list(itertools.accumulate(r_ks))
        total_pulls += R_k[k]
        S_k = ceil(S_ks[k]/eta)
        S_ks.append(S_k)
    chunk = ceil(data_size/total_pulls)
    return budget, chunk


class SuccessiveHalving:
    def __init__(self, arms: "OrderedDict[str, Arm]") -> None:
        self._arms = arms
        self._observer = ShiftObserver()
        # number of times an arm has been pulled
        self._pulls_performed = {arm_name: 0 for arm_name in self._arms}
        self._partial_results = {arm_name: {0: -1} for arm_name in self._arms}
        self._needed_pulls = {arm_name: [0] for arm_name in self._arms}
        for arm_name in self._arms:
            self._partial_results[arm_name][0] = self._arms[
                arm_name
            ].initial_error.num_errors

    def _successive_halving(
        self, budget: int, eta: int, arm_names: List[str] = None
    ) -> None:
        if arm_names is None:
            arm_names = list(self._arms.keys())

        abs_big_s_k = len(arm_names)
        big_b = budget
        big_r_k = 0
        log_value = max(1, ceil(log(abs_big_s_k, eta)))
        finished_arms = set()
        logger.info("log value: {}".format(log_value))
        for k in range(log_value):
            r_k = floor(big_b / (abs_big_s_k * log_value))
            logger.info("r_k: {}".format(r_k))
            if r_k == 0:
                break
            processes = []
            manager = multiprocessing.Manager()
            ret_value = manager.dict()
            logger.info("{} arms to test".format(len(arm_names)))
            for arm_name in arm_names:
                p = ArmProcess(
                    arm=self._arms[arm_name],
                    arm_name=arm_name,
                    big_r_k=big_r_k,
                    r_k=r_k,
                    observer=self._observer,
                    partial_results=self._partial_results[arm_name],
                    pulls_performed=self._pulls_performed[arm_name],
                    ret_value=ret_value,
                    len_arms=len(arm_names),
                    needed_pulls=self._needed_pulls[arm_name],
                )
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            logger.info("Finished a round...")

            for arm_name in arm_names:
                if arm_name not in ret_value:
                    continue
                if ret_value[arm_name]["finished"]:
                    finished_arms.add(arm_name)
                    continue
                self._pulls_performed[arm_name] = ret_value[arm_name]["pulls_performed"]
                self._partial_results[arm_name] = ret_value[arm_name]["partial_results"]
                self._needed_pulls[arm_name] = ret_value[arm_name]["needed_pulls"]
                self._arms[arm_name]._index = ret_value[arm_name]["index"]

            big_r_k += r_k
            arms_and_losses = []

            for arm_name in arm_names:
                logger.info("partial results for arm {}:{}".format(arm_name, self._partial_results[arm_name]))
                if arm_name not in finished_arms:
                    if big_r_k in self._partial_results[arm_name]:
                        arms_and_losses.append(
                            (arm_name, self._partial_results[arm_name][big_r_k])
                        )
                    else:
                        # now since we do not have defaultdict, we have to manually check
                        arms_and_losses.append((arm_name, -1))
            # if there are arms that cannot execute all pulls
            if -1 in map(lambda x: x[1], arms_and_losses):
                logger.info("losses: {} for big_r_k {}".format(arms_and_losses, big_r_k))
                logger.info("Cannot execute all pulls, exiting...")
                break
            sorted_arms_and_losses = sorted(arms_and_losses, key=lambda x: x[1])

            arms_and_losses_to_retain = sorted_arms_and_losses[: ceil(abs_big_s_k / eta)]
            arm_names = list(map(lambda x: x[0], arms_and_losses_to_retain))
            abs_big_s_k = ceil(abs_big_s_k / eta)
            if len(arm_names) == 0:
                logger.info("no more arms left, exiting...")
                break

    def run(self, eta: int, budget: int, jobs_db: JobsDB, job_hash: str):
        arms = list(self._arms.keys())
        current_budget = budget
        logger.info(
            "Running Successive Halving task {} with budget: {}".format(
                job_hash, current_budget
            )
        )
        self._successive_halving(budget=budget, eta=eta, arm_names=arms)
        logger.info("Saving results and exiting...")
        jobs_db.store_hyperband_job(job_hash)
