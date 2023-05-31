import numpy as np

from abc import ABC, abstractmethod

class LeverageSchedule(ABC):
    @abstractmethod
    def leverage(self, t: int, tau: float, init: float) -> float:
        pass

    def __call__(self, t: int, tau: float, init: float) -> float:
        return self.leverage(t, tau, init)

class ExactLeverageSchedule(LeverageSchedule):
    C: float

    def __init__(self, C: float) -> None:
        self.C = C

    def leverage(self, t: int, tau: float, init: float) -> float:
        return - np.log(tau / init) / (self.C * (2 ** (t + 1)))

class RelativeLeverageSchedule(LeverageSchedule):
    C: float

    def __init__(self, C: float) -> None:
        self.C = C

    def leverage(self, t: int, tau: float, init: float) -> float:
        return - np.log(tau / init) / (4 * self.C * t)