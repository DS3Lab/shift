from loguru import logger

from ._base import Observer, ProgressResult


class ShiftObserver(Observer):
    def on_update(self, name: str, progress_result: ProgressResult):
        logger.info(
            "Name: {}, Processed: {}, err: {}".format(
                name,
                progress_result.num_train_points_processed,
                progress_result.num_errors,
            )
        )
