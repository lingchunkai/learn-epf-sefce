from core import operations_gpu, util
from core.perfect_game import TreeState, TreeGame
from core.tree_game_properties import TreeGameProperties
from typing import List, Tuple
import torch
import numpy as np


class ValueAndStrategyExtractorGPU(object):
    def __init__(self,
                 tree_game: TreeGame,
                 tgp: TreeGameProperties,
                 network: torch.nn.Module,
                 device: torch.DeviceObjType,
                 colinearity_tol: float,
                 network_guarantees_concave: bool,
                 branching_factor: int,
                 points_per_state: int,
                 ):

        self.tree_game = tree_game
        self.tgp = tgp
        self.network = network
        self.colinearity_tol = colinearity_tol
        self.device = device
        self.network_guarantees_concave = network_guarantees_concave
        self.branching_factor = branching_factor
        self.points_per_state = points_per_state

        print('Moving feats to device:', device)
        all_feats = [x[0] for x in self.tgp.feats]
        self.feats_tensor = torch.stack(all_feats, dim=0).float().to(device)
        self.lb_tensor = torch.tensor(self.tgp.f_compete).float().to(device)
        self.ub_tensor = torch.tensor(self.tgp.f_coop).float().to(device)
        child_id_tensor, child_id_mask, terminal_tensor, threats_tensor, terminal_rews_tensor = self.preprocess_child_ids()
        self.child_id_tensor = child_id_tensor.to(self.device)
        self.child_id_mask = child_id_mask.to(self.device)
        self.terminal_tensor = terminal_tensor.to(self.device)
        self.threats_tensor = threats_tensor.to(self.device)
        self.terminal_rews_tensor = terminal_rews_tensor.to(self.device)

        self.ops = operations_gpu.OperationsGPU(self.device)

    def preprocess_child_ids(self):
        ID = torch.zeros((self.tree_game.num_infosets(),
                         self.branching_factor), dtype=torch.long)
        M = torch.ones((self.tree_game.num_infosets(),
                       self.branching_factor)) > 0
        is_terminal = torch.ones(self.tree_game.num_infosets()) < 0
        threats = torch.zeros(self.tree_game.num_infosets())
        terminal_rews = torch.zeros(
            (self.tree_game.num_infosets(), 2), dtype=torch.float)

        for parent_inf_id in range(self.tree_game.num_infosets()):
            parent_inf: TreeState = self.tree_game.infosets[parent_inf_id]

            assert self.branching_factor >= len(parent_inf.child_states)

            if parent_inf.is_terminal():
                is_terminal[parent_inf_id] = True
                rews = self.tree_game.infosets[parent_inf_id].rewards()
                terminal_rews[parent_inf_id, util.LEADER] = rews[util.LEADER]
                terminal_rews[parent_inf_id,
                              util.FOLLOWER] = rews[util.FOLLOWER]

            for child_idx, child_state in enumerate(parent_inf.child_states):
                ID[parent_inf_id, child_idx] = child_state.state_id
                if parent_inf.player_to_move() == util.FOLLOWER:
                    threats[child_state.state_id] = self.tgp.threats[child_state.state_id]
                else:
                    threats[child_state.state_id] = -float('inf')

            M[parent_inf_id][0: len(parent_inf.child_states)] = False

        return ID, M, is_terminal, threats, terminal_rews

    def get_coords_from_network(self,
                                inf_id: int,
                                ) -> Tuple[torch.tensor, torch.tensor]:
        """ Return (fol, lead), each tensors giving follower and leader coordinates.
            Each tensor is necessarily of the same size.
            Assumes that fol is non-increasing.

            Both fol, lead are of size `num_samples`, where `num_samples` is
            a parameter of self.network.
        """
        target_feats = self.feats_tensor[inf_id, :].unsqueeze(0)
        lb = self.lb_tensor[inf_id].unsqueeze(0)
        ub = self.ub_tensor[inf_id].unsqueeze(0)

        fol, lead = self.network(target_feats, lb, ub)
        fol, lead = fol[0, :], lead[0, :]

        return fol, lead

    def batch_get_coords_from_network(self,
                                      inf_ids: List[int],
                                      to_print=False,
                                      structural_loss=False
                                      ) -> Tuple[torch.tensor, torch.tensor]:
        """ Batch implmentation to the non-batched version which only calls the network once.
            Returns (fol, lead), each containng tensors giving follower and leader coordinates.

            Both fol, lead are of size len(inf_ids) x `num_samples` where `num_samples` is
            a parameter of sel.network.
        """
        target_feats = self.feats_tensor[inf_ids, :]
        lb = self.lb_tensor[inf_ids]
        ub = self.ub_tensor[inf_ids]

        net_out = self.network(target_feats, lb, ub)
        fol, lead, = net_out

        return fol, lead

    # =========== Computing Child Hulls and Coordinates ==============

    def compressed_envelope_from_hull(self, X, Y, mask):
        return self.ops.optimal_envelope(X, Y, mask, self.points_per_state)

    def batch_compute_children_hull(self,
                                    parent_inf_ids: List[int],
                                    trunc_ub: bool = False) -> Tuple[List[util.Hull], List[int]]:
        """ Note: trunc_ub only masks points greater than the truncation value,
            it does not perform interpolation!
        """

        nparents = len(parent_inf_ids)
        # Precompute reference indices first, may be jumbled in later steps!
        ref_indices = torch.arange(0, self.branching_factor).unsqueeze(1).expand(
            (-1, self.points_per_state)).reshape((1, -1)).unsqueeze(0).expand((nparents, -1, -1)).reshape((nparents, -1)).to(self.device)

        group_child_mask = self.child_id_mask[parent_inf_ids, :]
        group_points_mask = group_child_mask.unsqueeze(
            2).expand((-1, -1, self.points_per_state))
        flat_points_mask = group_points_mask.reshape(
            (-1, self.points_per_state))
        group_child_ids = self.child_id_tensor[parent_inf_ids, :]
        flat_child_ids = group_child_ids.flatten()
        flat_feats = self.feats_tensor[flat_child_ids, :]
        flat_lb = self.lb_tensor[flat_child_ids]
        flat_ub = self.ub_tensor[flat_child_ids]
        flat_X, flat_Y = self.network(flat_feats, flat_lb, flat_ub)

        # Modify values for terminal states to be singleton and set first point to not masked.
        flat_is_terminal_state = self.terminal_tensor[flat_child_ids]
        flat_points_mask[flat_is_terminal_state, 0] = False
        # NOTE: here we set all points in the row to be the value of
        # the terminal state. This is required because after replacing
        # the x values, they may not be in ascending order and this
        # would mess up future operations (e.g., truncate).
        # But we still mask all but the first one.
        flat_X[flat_is_terminal_state,
               :] = self.terminal_rews_tensor[flat_child_ids[flat_is_terminal_state], util.FOLLOWER].unsqueeze(1).expand((-1, self.points_per_state))
        flat_Y[flat_is_terminal_state,
               :] = self.terminal_rews_tensor[flat_child_ids[flat_is_terminal_state], util.LEADER].unsqueeze(1).expand((-1, self.points_per_state))

        # Truncate based on threat_val (child of leaders always have infinity threat)
        flat_X_trunc, flat_Y_trunc, flat_points_mask_trunc = self.ops.truncate(
            flat_X, flat_Y, self.threats_tensor[flat_child_ids], flat_points_mask)

        # Combine tensors together.
        comb_X_trunc = flat_X_trunc.reshape((nparents, -1))
        comb_Y_trunc = flat_Y_trunc.reshape((nparents, -1))
        comb_points_mask_trunc = flat_points_mask_trunc.reshape((nparents, -1))

        # Remove repeats.
        mask_repeats = self.ops.mask_of_repetitions(
            comb_X_trunc, comb_Y_trunc, comb_points_mask_trunc)
        comb_points_mask_trunc = torch.logical_or(
            comb_points_mask_trunc, mask_repeats)

        # Sort according to X locations.
        comb_X_trunc_sorted, sort_idx = torch.sort(comb_X_trunc, dim=1)
        comb_Y_trunc_sorted = torch.gather(comb_Y_trunc, 1, sort_idx)
        comb_points_mask_trunc_sorted = torch.gather(
            comb_points_mask_trunc, 1, sort_idx)
        ref_indices_sorted = torch.gather(ref_indices, 1, sort_idx)

        # Finally, run convex hull!
        X_, Y_, final_mask = self.ops.upper_concave_envelope(
            comb_X_trunc_sorted, comb_Y_trunc_sorted, comb_points_mask_trunc_sorted)
        final_mask = torch.logical_or(
            final_mask, comb_points_mask_trunc_sorted)

        # Truncate higher points higher than ub.
        if trunc_ub:
            mask_ub = X_ > self.ub_tensor[parent_inf_ids].unsqueeze(
                1).expand_as(X_)
            final_mask = torch.logical_or(final_mask, mask_ub)

        return X_, Y_, final_mask, ref_indices_sorted

    def extract_hull_and_act_list(self, inf_id):
        X, Y, final_mask, ref_indices = self.batch_compute_children_hull([
            inf_id], trunc_ub=True)
        X_ = X[~final_mask].flatten().detach().cpu().numpy()
        Y_ = Y[~final_mask].flatten().detach().cpu().numpy()
        ref_indices_ = ref_indices[~final_mask].flatten(
        ).detach().cpu().numpy()

        hull = np.zeros((X_.size, 2))
        hull[:, util.FOLLOWER] = X_
        hull[:, util.LEADER] = Y_

        return hull, ref_indices_

    def set_network(self, net):
        self.network = net
