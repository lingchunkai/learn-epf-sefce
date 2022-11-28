# Brute force solution for sefce for perfect information games

from core import util
from copy import deepcopy
from core.operations import concave_envelope, truncate, minkowski
from algs.tree_baselines import competitive
from policy.tabular_tree_policy import TabularTreePolicy
from typing import List, Tuple
import numpy as np


class Solver(object):
    def __init__(self, tree_game, colinearity_tol=0.0):
        self.tree_game = tree_game
        self.colinearity_tol = colinearity_tol

    def solve(self):
        self.init_lists()
        (_, _), self.threats = competitive(self.tree_game)
        self.backup()

        print('Root value function:', self.coords[0])

    def init_lists(self):
        self.coords = [None for i in range(len(self.tree_game.infosets))]

    def backup(self):
        for inf in reversed(self.tree_game.infosets):
            inf_id = inf.state_id
            state = inf.state
            if state.is_terminal():
                self.coords[inf_id] = \
                    np.atleast_2d((state.rewards()[util.LEADER],
                                  state.rewards()[util.FOLLOWER]))

            elif state.player_to_move() == util.LEADER:
                child_points_list = []
                for child_state in inf.child_states:
                    child_points_list.append(
                        self.coords[child_state.state_id]
                    )

                self.coords[inf_id], _ = concave_envelope(
                    child_points_list,
                    redundant_eps=self.colinearity_tol)

            elif state.player_to_move() == util.FOLLOWER:
                child_points = []
                for child_state in inf.child_states:
                    coords = self.coords[child_state.state_id]
                    threat = self.threats[child_state.state_id]
                    trunc_coords = np.array(truncate(coords, threat))
                    child_points.append(trunc_coords)

                self.coords[inf_id], _ = concave_envelope(
                    child_points,
                    redundant_eps=self.colinearity_tol)
            elif state.player_to_move() == util.CHANCE:
                child_points_list = []
                for child_state in inf.child_states:
                    child_points_list.append(self.coords[child_state.state_id])

                coords, right_end_action_idx = minkowski(
                    child_points_list, inf.probs, redundant_eps=self.colinearity_tol)
                self.coords[inf_id] = coords

    def tabular_policy(self, start_inf_id, init_target_follower_payoff):
        """ Get TabularTreePolicy by iterating over all infosets.

        TODO: use dict instead of list?
        """
        all_action_probs = [None] * len(self.tree_game.infosets)
        all_promised_payoffs = [None] * len(self.tree_game.infosets)

        L = [(start_inf_id, init_target_follower_payoff)]
        while len(L) > 0:
            inf_id, target_follower_payoff = L.pop()
            inf = self.tree_game.infosets[inf_id]

            if inf.is_terminal():
                act_probs = None
                promised_follower_rewards = None

            elif inf.player_to_move() == util.LEADER:
                child_points_list = []

                for child_state in inf.child_states:
                    child_points_list.append(self.coords[child_state.state_id])

                hull, act_ids = concave_envelope(
                    child_points_list, redundant_eps=self.colinearity_tol)

                act_probs, promised_follower_rewards = self.get_probs_and_follower_payoffs_from_hull(
                    hull, target_follower_payoff, act_ids, len(inf.child_states))

            elif inf.player_to_move() == util.FOLLOWER:
                child_points = []
                for child_state in inf.child_states:
                    coords = self.coords[child_state.state_id]
                    threat = self.threats[child_state.state_id]
                    trunc_coords = truncate(coords, threat)
                    child_points.append(trunc_coords)

                hull, act_ids = concave_envelope(
                    child_points, redundant_eps=self.colinearity_tol)

                act_probs, promised_follower_rewards = self.get_probs_and_follower_payoffs_from_hull(
                    hull, target_follower_payoff, act_ids, len(inf.child_states))

            elif inf.player_to_move() == util.CHANCE:
                list_coords = [self.coords[child_state.state_id]
                               for child_state in inf.child_states]
                hull, right_end_action_idx = minkowski(
                    list_coords, inf.probs, redundant_eps=self.colinearity_tol)
                act_probs = deepcopy(inf.probs)

                least_idx = util.find_idx_ge(
                    hull, target_follower_payoff, key=lambda x: x[util.FOLLOWER])

                if least_idx == 0:
                    # We give the minimum follower payoff for all child nodes.
                    promised_follower_rewards = [
                        x[0][util.FOLLOWER] for x in list_coords
                    ]
                else:
                    # Find for each child node, which part of the coords
                    # our answer should be in.
                    times_per_child_state = [0] * len(inf.child_states)
                    # These are the line segments which are fully included
                    for i in range(1, least_idx):
                        times_per_child_state[right_end_action_idx[i]] += 1

                    # Get follower payoffs accruing from full line segments.
                    promised_follower_rewards = []
                    for action_id in range(len(inf.child_states)):
                        promised_follower_rewards.append(
                            list_coords[times_per_child_state[action_id]])

                    # The last segment may be only partially included.
                    ratio = (target_follower_payoff - hull[least_idx-1][util.FOLLOWER]) / \
                        (hull[least_idx][util.FOLLOWER] -
                         hull[least_idx-1][util.FOLLOWER])
                    partial_act_idx = right_end_action_idx[least_idx]
                    partial_seg_idx = times_per_child_state[partial_act_idx] + 1
                    promised_follower_rewards += ratio * \
                        (list_coords[partial_act_idx][partial_seg_idx][util.FOLLOWER] -
                         list_coords[partial_act_idx][partial_seg_idx-1][util.FOLLOWER])

            all_action_probs[inf_id] = act_probs
            all_promised_payoffs[inf_id] = promised_follower_rewards

            for idx, child_inf in enumerate(inf.child_states):
                L.append((child_inf.state_id, promised_follower_rewards[idx]))
        return TabularTreePolicy(all_action_probs, all_promised_payoffs)

    def get_ideal_payoffs(self, inf_id):
        # Get peak leader payoff for some infoset.
        # TODO: use binary search

        coords = self.coords[inf_id]
        for i in range(len(coords)-1):
            g = util.grad(coords[i], coords[i+1])
            if g <= 0:
                return coords[i]

        return coords[-1]

    def get_probs_and_follower_payoffs_from_hull(self,
                                                 hull,
                                                 target_follower_payoff,
                                                 act_ids,
                                                 num_acts):
        """
        Note: for child nodes which are visited with 0 prob, we will require them to give a
        give the follow a payoff > -inf. This is merely a hack to make sure
        the follower payoff requirement can always be satisfied.

        When sampling or in actual play, these nodes will never be reached, unless the follower
        has deviated, in which case the leader will be playing a grim-strategy.
        """

        least_idx = util.find_idx_ge(
            hull[:, util.FOLLOWER], target_follower_payoff, key=lambda x: x)
        if least_idx == 0:
            probs = [1.0 if act_ids[least_idx] ==
                     act_id else 0.0 for act_id in range(num_acts)]
            follower_payoffs = [target_follower_payoff if act_ids[least_idx]
                                == act_id else -float('inf') for act_id in range(num_acts)]
        else:
            left_x = hull[least_idx-1][util.FOLLOWER]
            right_x = hull[least_idx][util.FOLLOWER]
            left_act = act_ids[least_idx-1]
            right_act = act_ids[least_idx]

            if left_act == right_act:
                # Mixing between same action.
                probs = [1.0 if left_act ==
                         act_id else 0.0 for act_id in range(num_acts)]
                follower_payoffs = [target_follower_payoff if left_act ==
                                    act_id else -float('inf') for act_id in range(num_acts)]
            else:
                # Mixing over different actions.
                ratio = (target_follower_payoff - left_x) / (right_x - left_x)
                probs, follower_payoffs = [0.0] * \
                    num_acts, [-float('inf')] * num_acts
                probs[left_act] = 1.0-ratio
                probs[right_act] = ratio

                follower_payoffs[left_act] = left_x
                follower_payoffs[right_act] = right_x
        return probs, follower_payoffs

    def check_duplicates_x(self, coords: List[Tuple[float, float]]):
        prev = None
        for x in coords:
            if x[util.FOLLOWER] == prev:
                return True
            prev = x[util.FOLLOWER]
        return False
