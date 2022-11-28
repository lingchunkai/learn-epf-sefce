from __future__ import annotations
from core.state import State
from typing import List, Optional
from core import util


class TreeState(object):
    def __init__(self, parent_state: TreeState,
                 state: State,
                 state_id: int,
                 probs: Optional[List[float]] = None):
        self.state_id = state_id
        self.parent_state = parent_state
        self.state = state
        self.child_states = []

        # Contains chance probabilities if this is a chance node.
        self.probs = probs

    def add_child_tree_state(self, child_tree_state) -> None:
        self.child_states.append(child_tree_state)

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def player_to_move(self):
        return self.state.player_to_move()

    def rewards(self):
        return self.state.rewards()


class TreeGame(object):
    def __init__(self, infosets: List[TreeState]):
        self.infosets = infosets

    def num_infosets(self) -> int:
        return len(self.infosets)


def convert_state_to_game(init_state: State) -> TreeGame:
    infoset_list = []
    dfs(init_state, None, infoset_list)

    return TreeGame(infoset_list)


def dfs(state: State,
        parent_tree_state: TreeState,
        infoset_list: List[TreeState]) -> None:

    if state.is_terminal():
        tree_state = TreeState(parent_tree_state, state, len(infoset_list))
        infoset_list.append(tree_state)
        if parent_tree_state is not None:
            parent_tree_state.add_child_tree_state(tree_state)
        return

    actions, probs = state.actions_and_probs()

    if state.player_to_move() == util.CHANCE:
        tree_state = TreeState(parent_tree_state, state,
                               len(infoset_list), probs=probs)
    else:
        tree_state = TreeState(parent_tree_state, state,
                               len(infoset_list), probs=None)
    infoset_list.append(tree_state)

    if parent_tree_state is not None:
        parent_tree_state.add_child_tree_state(tree_state)

    for a in actions:
        new_state = state.dup()
        new_state.next_state(a)

        dfs(new_state, tree_state, infoset_list)
