import torch
from core.perfect_game import TreeState
from games.treasure_hunt import TreasureHuntConfig, TreasureHuntState
from core import util
from typing import Tuple
from core.state import State


class FeaturizerBasic(object):
    def __init__(self, config: TreasureHuntConfig):
        self.config = config

    def cur_locs(self, state: TreasureHuntState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        cur_loc_pl1 = state.locs_pl1[-1]
        cur_loc_pl2 = state.locs_pl2[-1]

        return cur_loc_pl1, cur_loc_pl2

    def visited_map(self, state: TreasureHuntState) -> torch.Tensor:
        ''' Returns one hot tensor of size w*h on locations visited.
        '''
        f = torch.zeros((state.config.grid_h, state.config.grid_w))

        for y, x in state.locs_pl1:
            f[y, x] = 1.0

        for y, x in state.locs_pl2:
            f[y, x] = 1.0

        return f

    def player_to_move(self, state: TreasureHuntState) -> int:
        ''' Returns 1 for leader, 0 for follower, 2 for terminal.
        '''
        if state.player_to_move() is None:
            return -1
        return state.player_to_move()

    def num_moves_left(self, state: TreasureHuntState) -> int:
        '''
        '''
        return state.config.time_horizon - len(state.locs_pl2)

    def rewards_accumulated(self, state: TreasureHuntState) -> Tuple[float, float]:
        '''
        '''
        return state.get_accum_payoffs(util.LEADER), state.get_accum_payoffs(util.FOLLOWER)


class FeaturizerBruteForce(FeaturizerBasic):
    def __init__(self, config: TreasureHuntConfig, unpacked: bool = False):
        super().__init__(config)
        self.unpacked = unpacked

    def __call__(self, state: State):
        cur_loc_pl1, cur_loc_pl2 = self.cur_locs(state)
        visited_map = self.visited_map(state)
        p2move = self.player_to_move(state)
        num_moves_left = self.num_moves_left(state)
        rew_pl1, rew_pl2 = self.rewards_accumulated(state)

        f = torch.cat([torch.flatten(torch.tensor(cur_loc_pl1)),
                       torch.flatten(torch.tensor(cur_loc_pl2)),
                       torch.flatten(visited_map),
                       torch.flatten(torch.tensor([p2move])),
                       torch.flatten(torch.tensor([num_moves_left])),
                       torch.flatten(torch.tensor([rew_pl1])),
                       torch.flatten(torch.tensor([rew_pl2]))
                       ], dim=0)

        if not self.unpacked:
            return (f,)
        else:
            return f

    def feats_from_tree_state(self, tree_state: TreeState) -> torch.tensor:
        state: TreasureHuntState = tree_state.state

        return self.__call__(state)

    def feature_size(self):
        return 8 + self.config.grid_w * self.config.grid_h
