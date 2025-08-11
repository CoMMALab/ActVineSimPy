import math
import numpy as np

'''
math functions regarding rotations
'''

def wrap_rotation(diff):
    """
    Wrap the angle into [-pi, pi).
    """
    
    # Wrap diff into (-pi, pi]
    return (diff + np.pi) % (2 * np.pi) - np.pi

def shortest_rotation(angle1, angle2):
    diff = angle2 - angle1
    return wrap_rotation(diff)

def test_shortest_rotation():
    assert math.isclose(shortest_rotation(0, 0), 0)
    assert math.isclose(shortest_rotation(0, math.pi), -math.pi)
    assert math.isclose(shortest_rotation(0, -math.pi), -math.pi)
    assert math.isclose(shortest_rotation(0, 2 * math.pi), 0)
    assert math.isclose(shortest_rotation(0, -2 * math.pi), 0)
    assert math.isclose(shortest_rotation(math.pi / 2, -math.pi / 2), -math.pi)
    assert math.isclose(shortest_rotation(-math.pi / 2, math.pi / 2), -math.pi)
    
if __name__ == "__main__":
    test_shortest_rotation()
    print("All tests passed.")