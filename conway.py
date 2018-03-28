"""
Simple Conway's game of life on the Sense Hat
"""

import asyncio
import signal
import sys
from colorsys import hsv_to_rgb
from math import floor

import numpy as np
from scipy.signal import convolve2d

try:
    from sense_hat import SenseHat
except ImportError:
    class SenseHat:
        """
        Dummy class to run on dev machine
        """

        def __getattr__(self, item):
            return lambda *x, **y: None

        def get_compass(self):
            return np.random.uniform(0, 360)

Y_SIZE, X_SIZE = 100, 100
Y_VIEW, X_VIEW = floor(Y_SIZE / 2), floor(X_SIZE / 2)
THRESHOLD = round(Y_SIZE * X_SIZE * 0.08)

a = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=np.uint8)


class BoringException(Exception):
    pass


class PeriodicityException(BoringException):
    pass


class EmptyException(BoringException):
    pass


class NothingInViewException(EmptyException):
    pass


class LongException(BoringException):
    pass


def new_seed() -> np.ndarray:
    """
    New random seed
    :return:
    """
    return np.random.randint(0, 2, (Y_SIZE, X_SIZE), dtype=np.uint8)


def get_color(num: float) -> np.ndarray:
    return np.around(np.array(hsv_to_rgb(num, 1, 1)) * 255).astype(np.int8)


def new_color() -> np.ndarray:
    """
    New random color with equal saturation and value

    Using hsv with saturation and value equal to randomize to colors
    of the same brightness on the LED matrix.
    :return:
    """
    return get_color(np.random.uniform())


def to_rgb565(color: np.ndarray) -> np.ndarray:
    """
    RGB888 tuple to RGB565
    :param color:
    :return:
    """
    return color >> np.array([3, 2, 3]) & np.array([0x1f, 0x3f, 0x1f])


def color_state(s: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    make state into list of color
    :param s:
    :param c:
    :return:
    """
    s = s.reshape((-1, 1)).repeat(3, axis=1)
    s[s[:, 0] == 1, :] = c
    return s


def life_step(state: np.ndarray) -> np.ndarray:
    count = convolve2d(state, a, mode='same', boundary='wrap')
    return ((count == 3) | (state & (count == 2))).astype(np.uint8)


# np.random.seed(0)

async def main(sense: SenseHat):
    seed = new_seed()
    color = new_color()

    generation = 0

    def reset():
        nonlocal generation, color, seed
        generation = 0
        seed = new_seed()
        c = new_color()
        # the average distance between 2 points in the rgb565 colors is about 29.28
        while np.linalg.norm(to_rgb565(c) - to_rgb565(color)) < 30:
            c = new_color()
        color = c
        sense.clear()

    while True:
        sl = asyncio.sleep(0.1)

        state = life_step(seed)

        try:
            s = state[Y_VIEW - 4:Y_VIEW + 4, X_VIEW - 4:X_VIEW + 4]
            sense.set_pixels(color_state(s, color))

            if not s.any():
                raise NothingInViewException("Nothing in view")

            if not state.any():
                raise EmptyException("Empty")

            if (seed ^ state).sum() < THRESHOLD:
                raise LongException("Too few changes")
        except BoringException as e:
            print(e)
            await sl
            await asyncio.sleep(2)
            reset()
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
