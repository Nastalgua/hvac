from typing import Tuple

TEMP_FACTOR = -1

class AC:
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.on = False
        self.factor = TEMP_FACTOR
