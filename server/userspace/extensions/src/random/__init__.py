import os
import random

from extensions.extension import ShiftExtension


class RandomSearchExtension(ShiftExtension):
    def __init__(self):
        super().__init__("RandomSearch")

    def __call__(self, models, datasets):
        """
        We should allow the extensions to run a shell command, such that these extensions does not have to be written in Python, but also in other languages, etc.
        """
        return random.shuffle(models)
