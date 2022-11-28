from __future__ import annotations
from core.state import State
from core import util
from copy import deepcopy
from typing import List, Optional, Tuple, Sequence
import random
import numpy as np


def generate_grid(grid_w: int,
                  grid_h: int,
                  ls: float,
                  sd: float,
                  seed: int):

    grid_size = grid_w * grid_h
    Y, X = np.unravel_index(list(range(grid_size)), (grid_h, grid_w))

    def sq_exp_kernel(a: np.ndarray,
                      b: np.ndarray,
                      length_scale: float,
                      sd: float):
        return sd**2 * np.exp(-np.sum((a - b)**2)/2/length_scale**2)

    # Compute K
    K = np.zeros((grid_size, grid_size))
    for a_1d in range(grid_size):
        for b_1d in range(grid_size):
            a_coord = np.array([Y[a_1d], X[a_1d]])
            b_coord = np.array([Y[b_1d], X[b_1d]])
            K[a_1d, b_1d] = sq_exp_kernel(a_coord, b_coord, ls, sd)

    # Sample from multivariate G.
    gen = np.random.default_rng(seed=seed)
    sample = gen.multivariate_normal(mean=np.zeros(grid_size), cov=K)
    sample = sample.reshape((grid_h, grid_w))
    return np.exp(sample)


class TreasureHuntConfig(object):
    def __init__(self,
                 rew_pl1: np.ndarray,
                 rew_pl2: np.ndarray,
                 time_horizon: int):
        self.grid_h, self.grid_w = rew_pl1.shape
        self.rew_pl1 = rew_pl1
        self.rew_pl2 = rew_pl2
        self.time_horizon = time_horizon
        assert rew_pl1.shape == rew_pl2.shape


class TreasureHuntState(State):
    def __init__(self,
                 config: TreasureHuntConfig,
                 locs_pl1,
                 locs_pl2,
                 ):

        self.config = config
        self.locs_pl1 = locs_pl1
        self.locs_pl2 = locs_pl2

        self.rew_pl1, self.rew_pl2 = None, None

    @staticmethod
    def init_state(config: TreasureHuntConfig):
        center = (config.grid_h//2, config.grid_w//2)
        return TreasureHuntState(config, [center], [center])

    def is_terminal(self):
        return len(self.locs_pl2) == self.config.time_horizon

    def player_to_move(self):
        if self.is_terminal():
            return None
        if len(self.locs_pl2) < len(self.locs_pl1):
            return util.PLAYER2
        elif len(self.locs_pl2) == len(self.locs_pl1):
            return util.PLAYER1
        else:
            assert False

    def next_state(self, action):
        p = self.player_to_move()

        if p == util.PLAYER1:
            next_loc = self.next_loc(self.locs_pl1[-1], action)
            self.locs_pl1.append(next_loc)
        else:
            next_loc = self.next_loc(self.locs_pl2[-1], action)
            self.locs_pl2.append(next_loc)

    def actions_and_probs(self):
        return ['u', 'd', 'l', 'r', '-'], None

    def rewards(self) -> Tuple[float, float]:
        if not self.is_terminal():
            return None
        else:
            rew_pl1 = self.get_accum_payoffs(util.PLAYER1)
            rew_pl2 = self.get_accum_payoffs(util.PLAYER2)

            return (rew_pl1, rew_pl2)

    def __str__(self):
        return 'Locs pl1: ' + str(self.locs_pl1) + '\n' + \
            'Locs pl2: ' + str(self.locs_pl2)

    def __repr__(self):
        return self.__str__()

    def dup(self):
        return TreasureHuntState(self.config,
                                 deepcopy(self.locs_pl1),
                                 deepcopy(self.locs_pl2))

    def next_loc(self, cur_loc, action):
        next_loc = list(cur_loc)
        if action == 'u':
            if cur_loc[0] > 0:
                next_loc[0] -= 1
        elif action == 'd':
            if cur_loc[0] < self.config.grid_h-1:
                next_loc[0] += 1
        elif action == 'l':
            if cur_loc[1] > 0:
                next_loc[1] -= 1
        elif action == 'r':
            if cur_loc[1] < self.config.grid_w-1:
                next_loc[1] += 1
        elif action == '-':
            pass
        return tuple(next_loc)

    def get_accum_payoffs(self, pid) -> float:
        if self.rew_pl1 is None:
            self.rew_pl1 = self.get_cum_rewards(
                [self.locs_pl1, self.locs_pl2], self.config.rew_pl1)
            self.rew_pl2 = self.get_cum_rewards(
                [self.locs_pl1, self.locs_pl2], self.config.rew_pl2)

        if pid == util.PLAYER1:
            return self.rew_pl1
        elif pid == util.PLAYER2:
            return self.rew_pl2
        else:
            assert False

    def get_cum_rewards(self, locs_lists: List[List], rew_grid) -> float:
        t = 0.

        # TODO: set criterion to either use set or list to extract rewards.
        use_set = True

        if use_set == True:
            d = set()
            for locs in locs_lists:
                for (y, x) in locs:
                    if (y, x) not in d:
                        t += rew_grid[y, x]
                        d.add((y, x))
            return t
        else:
            visited = [[False for i in range(self.config.grid_w)]
                       for j in range(self.config.grid_h)]
            for locs in locs_lists:
                for (y, x) in locs:
                    if not visited[y][x]:
                        t += rew_grid[y, x]
                        visited[y][x] = True
            return t


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    leader_payoff = generate_grid(7, 7, 2, 0.1, 142857)
    follower_payoff = generate_grid(7, 7, 2, 0.1, 428571)

    config = TreasureHuntConfig(leader_payoff, follower_payoff, 5)
    s = TreasureHuntState.init_state(config)

    while(not s.is_terminal()):
        # Take random action
        acts, probs = s.actions_and_probs()

        if s.player_to_move() == util.CHANCE:
            act_id = random.choices(range(len(acts)), weights=probs)[0]
        else:
            act_id = random.randint(0, len(acts)-1)
        action = acts[act_id]

        s.next_state(action)

        print(s)
        print('---')

    print(s.rewards())
