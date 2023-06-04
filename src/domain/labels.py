from enum import unique, Enum

@unique
class Label(Enum):
    """
    Классы на изображениях.
    """
    PLANE = 0
    CAR = 1
    BIRD = 2
    CAT = 3
    DEER = 4
    DOG = 5
    FROG = 6
    HORSE = 7
    SHIP = 8
    TRUCK = 9
