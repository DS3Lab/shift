import multiprocessing
from typing import Dict, List

from loguru import logger
from optimizations.observers import ShiftObserver

from .interface import ShiftArm


class ArmProcess(multiprocessing.Process):
    def __init__(
        self,
        arm: ShiftArm,
        arm_name: str,
        big_r_k: int,
        r_k: int,
        observer: ShiftObserver,
        partial_results: Dict,
        pulls_performed: int,
        ret_value: Dict,
        len_arms: int,
        needed_pulls: List,
    ):
        multiprocessing.Process.__init__(self)
        self.big_r_k = big_r_k
        self.r_k = r_k
        self.arm = arm
        self.arm_name = arm_name
        self.partial_results = partial_results
        self.pulls_performed = pulls_performed
        self.observer = observer
        self.ret_value = ret_value
        self.len_arms = len_arms
        self.needed_pulls = needed_pulls

    def run(self):
        target_num_pulls = self.big_r_k + self.r_k
        pulls_performed = self.pulls_performed
        if pulls_performed < target_num_pulls:
            logger.info(
                "Arm {} needs to be pull {} times".format(
                    self.arm_name, target_num_pulls
                )
            )
            logger.info(
                "[{}]: arm size: {}, arm index: {}, r_k: {}, pulls_performed: {}, target_num_pulls: {}".format(
                    self.arm_name,
                    self.arm.total_size,
                    self.arm._index,
                    self.r_k,
                    pulls_performed,
                    target_num_pulls,
                )
            )
            if self.len_arms == 1:
                self.ret_value[self.arm_name] = {
                    # If there's only one left, it is considered finished automatically
                    "finished": True,
                    "pulls_performed": self.pulls_performed,
                    "partial_results": self.partial_results,
                    "index": self.arm._index,
                    "needed_pulls": self.needed_pulls,
                }
            elif self.arm.can_progress():
                self.needed_pulls.append(self.r_k)
                logger.info("Arm {} can progress".format(self.arm_name))

                current_progress = self.arm.progress(self.needed_pulls)
                
                self.observer.on_update(self.arm.name, current_progress)
                self.partial_results[target_num_pulls] = current_progress.num_errors
                logger.info("Arm {}, target_num_pulls: {}, partial results: {}".format(self.arm.name,target_num_pulls, self.partial_results))
                self.ret_value[self.arm_name] = {
                    "pulls_performed": self.pulls_performed,
                    "partial_results": self.partial_results,
                    "finished": not self.arm.can_progress(),
                    "index": self.arm._index,
                    "needed_pulls": self.needed_pulls,
                }
            else:
                self.ret_value[self.arm_name] = {
                    "finished": True,
                    "pulls_performed": self.pulls_performed,
                    "partial_results": self.partial_results,
                    "index": self.arm._index,
                    "needed_pulls": self.needed_pulls,
                }
        else:
            self.ret_value[self.arm_name] = {
                "finished": False,
                "partial_results": self.partial_results,
                "pulls_performed": self.pulls_performed,
                "index": self.arm._index,
                "needed_pulls": self.needed_pulls,
            }
