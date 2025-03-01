from enum import Enum

class RewardType(Enum):
    """
    Defines the types of reward functions available in the environment.

    Attributes:
        PROFIT: Rewards based solely on selling surplus energy.
        INTERNAL_DEMAND: Rewards prioritizing internal demand before selling energy.
    """
    PROFIT = "profit"
    INTERNAL_DEMAND = "internal"


class DemandType(Enum):
    """
    Defines the types of demand functions available in the environment.

    Attributes:
        GAUSSIAN: Models demand using two overlapping Gaussian distributions.
        SINUSOIDAL: Implements a combination of sine waves with different frequencies.
        STEP: Uses discrete jumps in demand at specific time intervals.
    """
    GAUSSIAN = "gaussian"
    SINUSOIDAL = "sinusoidal"
    STEP = "step"