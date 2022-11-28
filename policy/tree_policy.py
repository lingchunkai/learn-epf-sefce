from abc import ABC, abstractmethod, abstractstaticmethod


class TreePolicy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_probs(self, infoset_id):
        pass
