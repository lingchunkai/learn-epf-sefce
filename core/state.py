from __future__ import annotations
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Any, List, Tuple


class State(ABC):
    def __init__(self):
        pass

    @abstractstaticmethod
    def init_state(self) -> State:
        pass

    @abstractmethod
    def next_state(self, action) -> State:
        pass

    @abstractmethod
    def actions_and_probs(self) -> Tuple[List[Any], List[float]]:
        pass

    def actions(self) -> List[Any]:
        return self.actions_and_probs()[0]

    @abstractmethod
    def rewards(self) -> Tuple[float | int, float | int]:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def dup(self) -> State:
        pass
