"""
Simple Conway's game of life on the Sense Hat
"""

import asyncio
import signal
import sys
from collections import deque
from colorsys import hsv_to_rgb

import numpy as np

try:
    from sense_hat import SenseHat
except ImportError:
    class SenseHat:
        """
        Dummy class to run on dev machine
        """

        def __getattr__(self, item):
            return lambda *x, **y: None

Y_SIZE, X_SIZE = 8, 8


class BoringException(Exception):
    pass


class PeriodicityException(BoringException):
    pass


class EmptyException(BoringException):
    pass


class LongException(BoringException):
    pass


def new_seed() -> np.ndarray:
    """
    New random seed
    :return:
    """
    return np.random.randint(0, 2, (Y_SIZE, X_SIZE), dtype=np.uint8)


def new_color() -> np.ndarray:
    """
    New random color with equal saturation and value

    Using hsv with saturation and value equal to one prevents from randomizing to
    have colors of the same brightness on the LED matrix.
    :return:
    """
    return np.around(np.array(hsv_to_rgb(np.random.uniform(), 1, 1)) * 255)


# np.random.seed(0)

async def main(sense):
    prev_seeds = deque(maxlen=30)

    # seed = np.random.choice(2, (y_size, x_size), True)
    seed = new_seed()
    color = new_color()

    generation = 0

    while True:
        sl = asyncio.sleep(0.2)

        prev_seeds.append(seed)

        p_seed = np.pad(seed, 1, mode='wrap')

        state = np.zeros((Y_SIZE, X_SIZE), dtype=np.uint8)

        it = np.nditer(seed, flags=['multi_index'], op_flags=['readonly'])

        while not it.finished:
            y, x = it.multi_index
            score = p_seed[y:y + 3, x:x + 3].sum() - it[0]
            if (not it[0] and score == 3) or (it[0] and 2 <= score < 4):
                state[y, x] = 1
            it.iternext()

        s = state.reshape((-1, 1)).repeat(3, axis=1)
        s[s[:, 0] == 1, :] = color
        sense.set_pixels(s.tolist())

        try:
            if not state.sum():
                raise EmptyException()

            for p in reversed(prev_seeds):
                if (p == state).all():
                    raise PeriodicityException()

            if generation > 300:
                raise LongException()
        except BoringException:
            await asyncio.sleep(3)
            generation = 0
            seed = new_seed()
            color = new_color()
            prev_seeds.clear()
            continue

        seed = state
        generation += 1
        await sl


if __name__ == "__main__":
    sense = SenseHat()


    def handle_sigint(signum, frame):
        """
        clear LED matrix
        :param signum:
        :param frame:
        :return:
        """
        sense.clear()
        sys.exit(0)


    signal.signal(signal.SIGINT, handle_sigint)

    loop = asyncio.get_event_loop()

    loop.run_until_complete(main(sense))
