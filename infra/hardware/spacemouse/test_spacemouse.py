import time
import numpy as np
from infra.hardware.spacemouse.spacemouse_expert import SpaceMouseExpert

def test_spacemouse():
    spacemouse0 = SpaceMouseExpert()
    with np.printoptions(precision=3, suppress=True):
        while True:
            action, buttons = spacemouse0.get_action()
            print(f"Spacemouse action: {action}, buttons: {buttons}")
            time.sleep(0.1)


def main():
    """Call spacemouse test."""
    test_spacemouse()


if __name__ == "__main__":
    main()
