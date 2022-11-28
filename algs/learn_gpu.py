from __future__ import annotations
from core.operations_gpu import OperationsGPU
import torch
from core.perfect_game import TreeState, TreeGame
import random
from core import util
import matplotlib.pyplot as plt
from algs.sefce_perfect_solver import Solver
from typing import List
from copy import deepcopy
import os
import sacred
import numpy as np
from core.tree_game_properties import TreeGameProperties
from core.value_and_strategy_extractor_gpu import ValueAndStrategyExtractorGPU
from algs.hull_loss_gpu import HullLoss

plt.rcParams["figure.figsize"] = (12.5, 12.5)


class LearnerGPU(object):
    def __init__(self,
                 tree_game: TreeGame,
                 featurizer,
                 network: torch.nn.Module,
                 loss_type: str,
                 device,
                 points_per_state,
                 network_guarantees_concave,
                 colinearity_tol=0.0,
                 monotonic=False,
                 learn_from_ground_truth: bool = False,
                 expt: sacred.Experiment = None):

        self.tree_game = tree_game

        print('Processing game tree and features')
        self.tgp = TreeGameProperties(tree_game, featurizer)
        self.tgp.preprocess()
        self.expt = expt
        self.device = device
        self.monotonic = monotonic
        self.learn_from_ground_truth = learn_from_ground_truth
        self.loss_type = loss_type
        self.colinearity_tol = colinearity_tol
        self.points_per_state = points_per_state

        self.ops = OperationsGPU(self.device)

        self.network_guarantees_concave = network_guarantees_concave
        self.main_network = network

        self.main_network_extractor = \
            ValueAndStrategyExtractorGPU(tree_game,
                                         self.tgp,
                                         self.main_network,
                                         self.device,
                                         colinearity_tol,
                                         network_guarantees_concave,
                                         self.tgp.branching_factor,
                                         points_per_state,
                                         )

        self.frozen_network_extractor = deepcopy(self.main_network_extractor)

        self.update_frozen_network()

    def learning(self,
                 num_iters: int = 10000000,
                 lr: float = 1e-5,
                 update_frozen_network_freq=50,
                 batch_size=32,
                 save_plot_frequency: int = 100,
                 save_file_prefix='',
                 sampling_method='uniform_state',
                 fixed_indices=None,
                 device=torch.device('cpu')):

        self.solver = Solver(
            self.tree_game, colinearity_tol=self.colinearity_tol)
        self.solver.solve()

        reach_probs = self.precompute_uniform_reach_probabilities(
            zero_on_terminal=True)
        state_sampler = torch.distributions.Categorical(
            torch.tensor(reach_probs))
        reach_probs = [reach_probs[j] for j in self.tgp.inf_ids_non_term]
        s = sum(reach_probs)
        reach_probs = [reach_probs[j]/s for j in range(len(reach_probs))]

        self.update_frozen_network()

        optim = torch.optim.Adam(
            self.main_network.parameters(), lr=lr, amsgrad=True)
        # optim = torch.optim.SGD(self.main_network.parameters(), lr=lr)
        # optim = torch.optim.RMSprop(self.network.parameters(), lr=lr)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1000)

        bsize = min([batch_size, len(self.tgp.inf_ids_non_term)])
        if fixed_indices is not None:
            my_fixed_indices = fixed_indices

        # Main outer loop.
        for it in range(num_iters):
            # --- Get training data indices ---
            if fixed_indices is None:
                if sampling_method == 'uniform_state':
                    inf_ids = random.sample(self.tgp.inf_ids_non_term, bsize)
                elif sampling_method == 'uniform_traj':
                    inf_ids = state_sampler.sample((bsize,))
            else:
                assert sampling_method == 'uniform_state'
                inf_ids = random.sample(my_fixed_indices, bsize)

            # --- Compute differentiable losses ---
            # Get hulls of which we want to update, by running it through self.network.
            X, Y = self.main_network_extractor.batch_get_coords_from_network(
                inf_ids, it > 10000, structural_loss=False)
            X_target, Y_target, target_mask = self.compute_training_hulls(
                inf_ids)

            L_ = self.compute_loss(X_target, Y_target, X,
                                   Y, target_mask=target_mask)
            L = torch.sum(L_)

            print('Iter %d Loss' % (it, ),
                  L.cpu().detach().numpy().item(), '\t',
                  torch.max(L_))

            # --- Checkpointing ---
            if it % (save_plot_frequency * 20) == 0:
                fname = save_file_prefix + 'model%d.pt' % (it, )
                torch.save(self.main_network, fname)
                self.expt.add_artifact(fname)
                os.remove(fname)

                fname = save_file_prefix + 'checkpoint%d.pt' % (it, )
                torch.save({
                    'iter': iter,
                    'model_state_dict': self.main_network.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                }, fname)
                self.expt.add_artifact(fname)
                os.remove(fname)
            # --- Monitoring ---
            if it % save_plot_frequency == 0:
                fname = save_file_prefix + 'Learned%d.png' % (it,)
                self.plot_network_values_for_infoset(
                    0, save_name=fname, save=True)

                """
                for k in self.tgp.inf_ids_non_term:
                    fname = save_file_prefix + 'Learned_%d_%d.png' % (k, it,)
                    self.plot_network_values_for_infoset(
                        k, save_name=fname, save=True)
                """

            # We only do a full eval if the number of states are too high
            # GPU runs out of memory.
            # TODO: speed up by splitting into small batches.
            if it % (save_plot_frequency * 20) == 0 and len(self.tgp.inf_ids_non_term) < 5000:
                culprits_fname = save_file_prefix + 'Worst%d.png' % (it, )
                loss_hist_fname = save_file_prefix + \
                    'LossHist%d.png' % (it, )
                self.full_eval(worst_culprits_fname=culprits_fname,
                               loss_histogram_fname=loss_hist_fname,
                               save=True)

            # --- Perform gradient steps ---
            L.backward()

            optim.step()
            optim.zero_grad()
            # scheduler.step()

            # --- Perform book-keeping ---
            # slow update if later iteration, finetuning.
            if it > 10000 and it % update_frozen_network_freq == 0:
                self.update_frozen_network()
            elif it < 10000 and it % 500 == 0:  # fast update at the start.
                self.update_frozen_network()

    def compute_training_hulls(self, inf_ids, gt=None):
        """
        Get hulls of children either from ground truth, or
        by running through network and stiching.
        """
        if gt is None:
            gt = self.learn_from_ground_truth

        if gt == False:
            with torch.no_grad():
                X, Y, mask, _ = self.frozen_network_extractor.batch_compute_children_hull(
                    inf_ids)
        else:
            # Train against ground truth.
            # TODO: speed up, this was not designed for GPU.
            computed_hulls = []
            for inf_id in inf_ids:
                # computed_hulls.append(self.list_of_tuples_to_hull(
                #     self.solver.coords[inf_id]))
                computed_hulls.append(self.solver.coords[inf_id])

            largest_size = max([x.shape[0] for x in computed_hulls])
            for i, c in enumerate(computed_hulls):
                computed_hulls[i] = np.pad(
                    c, ((largest_size - c.shape[0], 0), (0, 0)), mode='edge')

            hulls = np.stack(computed_hulls, axis=0)
            X = torch.tensor(hulls[:, :, util.FOLLOWER]).to(self.device)
            Y = torch.tensor(hulls[:, :, util.LEADER]).to(self.device)
            mask = torch.ones_like(X) < 0

        # Adjust computed hulls to only decreasing part of the hull
        if self.monotonic:
            mask_dec = self.ops.extract_decreasing(X, Y, mask)
            mask = torch.logical_or(mask, mask_dec)

        X_, Y_ = X, Y
        return X_, Y_, mask

    # =========== Monitoring ==============

    def full_eval(self,
                  worst_culprits_fname: str = None,
                  loss_histogram_fname: str = None,
                  save: bool = True):
        '''
        Compute losses over all non-terminal states and plot the top
        16 culprint states.
        '''
        # Find the exact losses for all non-terminal states.
        inf_ids = self.tgp.inf_ids_non_term
        X, Y = self.main_network_extractor.batch_get_coords_from_network(
            inf_ids, False, structural_loss=False)

        target_X, target_Y, target_mask = self.compute_training_hulls(inf_ids)

        L_ = self.compute_loss(target_X, target_Y, X, Y,
                               target_mask=target_mask, mask=None)

        # Get worst culprints
        # Sort contribution in terms of states and plot the
        # top 16 biggest contributions.
        sorted_losses, ind = torch.sort(L_, descending=True)
        for z in range(16):
            plt.subplot(4, 4, z+1)
            id = ind[z]
            plt.plot(X[id, :].cpu().detach().numpy(),
                     Y[id, :].cpu().detach().numpy(), '-o')
            target_X_ = target_X[id, :][~target_mask[id, :]]
            target_Y_ = target_Y[id, :][~target_mask[id, :]]
            plt.plot(target_X_.cpu().detach().numpy(),
                     target_Y_.cpu().detach().numpy(), '-x')
            plt.plot([x[util.FOLLOWER] for x in self.solver.coords[self.tgp.inf_ids_non_term[id]]],
                     [x[util.LEADER] for x in self.solver.coords[self.tgp.inf_ids_non_term[id]]], '-x')
            plt.title("%d: %.4f" %
                      (inf_ids[ind[z]], sorted_losses[z].cpu().detach().numpy().item()))

        # Save or display on screen.
        if save:
            plt.savefig(worst_culprits_fname)
            plt.clf()
            if self.expt is not None:
                self.expt.add_artifact(worst_culprits_fname)
                os.remove(worst_culprits_fname)
        else:
            plt.plot()

        # Get histogram of errors
        plt.hist(sorted_losses.cpu().detach().numpy(), bins=500)
        if save:
            plt.savefig(loss_histogram_fname)
            plt.clf()
            if self.expt is not None:
                self.expt.add_artifact(loss_histogram_fname)
                os.remove(loss_histogram_fname)
        else:
            plt.plot()

    def plot_network_values_for_infoset(self,
                                        inf_id: int,
                                        save_name: str = None,
                                        save: bool = True) -> None:
        with torch.no_grad():
            # Plot predicted value
            X, Y = self.main_network_extractor.get_coords_from_network(
                inf_id)
            plt.plot(X.cpu().numpy(), Y.cpu().numpy(), '-o')

            # Plot one-step-lookahead.
            target_X, target_Y, mask = self.compute_training_hulls(
                [inf_id], gt=False)
            target_X = target_X[0, ~mask[0, :]]
            target_Y = target_Y[0, ~mask[0, :]]

            plt.plot(target_X.cpu(),
                     target_Y.cpu(), '-+')

            # Plot true values.
            plt.plot([x[util.FOLLOWER] for x in self.solver.coords[inf_id]],
                     [x[util.LEADER] for x in self.solver.coords[inf_id]], '-x')

            plt.legend(['Predicted', 'Lookahead', 'True'])

            if save:
                plt.savefig(save_name)
                plt.clf()
                if self.expt is not None:
                    self.expt.add_artifact(save_name)
                    os.remove(save_name)
            else:
                plt.plot()

    def plot_children_and_parent_comp(self, inf_id, fname):
        child_ids = [
            x.state_id for x in self.tree_game.infosets[inf_id].child_states]

        child_fol, child_lead = self.frozen_network_extractor.batch_get_coords_from_network(
            child_ids
        )

        direct_fol, direct_lead = self.main_network_extractor.get_coords_from_network(
            inf_id)
        lookahead_fol, lookahead_lead, mask, _ = self.frozen_network_extractor.batch_compute_children_hull([
                                                                                                           inf_id])
        lookahead_fol = lookahead_fol.detach().cpu().numpy()
        lookahead_lead = lookahead_lead.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        plt.subplot(1, 2, 1)
        X, Y = [], []
        for j in range(lookahead_fol.shape[1]):
            if mask[0][j] == False:
                X.append(lookahead_fol[0][j])
                Y.append(lookahead_lead[0][j])
        print('X lookahead', X)
        print('Y lookahead', X)
        plt.plot(X, Y, '-o')
        plt.plot(direct_fol.detach().cpu().numpy(),
                 direct_lead.detach().cpu().numpy(), '-x')

        plt.subplot(1, 2, 2)
        c_fol = child_fol.detach().cpu().numpy()
        c_lead = child_lead.detach().cpu().numpy()
        for i in range(child_fol.shape[0]):
            plt.plot(c_fol[i, :], c_lead[i, :], '-x')

        plt.savefig(fname)
        plt.clf()
        if self.expt is not None:
            self.expt.add_artifact(fname)
            os.remove(fname)

    # =========== Loss Functions ==============

    def compute_loss(self, target_fol, target_lead,
                     fol, lead, target_mask=None, mask=None):
        bsize = target_fol.size()[0]
        if self.loss_type == 'points':
            L_ = self.get_points_dist_loss(
                target_fol, target_lead, fol, lead,
                type='l2',
                target_mask=target_mask,
                mask=mask,
                matching=False,  # TODO: change depending on what we need!
                use_weights=False)  # use decay?
        elif self.loss_type == 'traditional':
            raise NotImplementedError()
        elif self.loss_type == 'dual_points_l2':
            raise NotImplementedError()
        else:
            assert False, 'Invalid loss type.'

        return L_

    def get_points_dist_loss(self,
                             target_fol,
                             target_lead,
                             fol,
                             lead,
                             type='l2',
                             target_mask=None,
                             mask=None,
                             use_weights=False,
                             matching=False,
                             verbose=False):
        if type == 'l2':
            if mask is None:
                mask = torch.ones_like(fol) < 0
            if target_mask is None:
                target_mask = torch.ones_like(target_fol) < 0

            X1, Y1, mask1 = self.ops.shift_masked_to_right(
                target_fol, target_lead, target_mask, [])
            X2, Y2, mask2 = self.ops.shift_masked_to_right(
                fol, lead, mask, [])
            hl = HullLoss(self.device)
            return hl.l2_norm(X1, Y1, X2, Y2, mask1, mask2)
            # xdiff = torch.sum((target_fol - fol)**2, dim=1)
            # ydiff = torch.sum((target_lead - lead)**2, dim=1)
        elif type == 'l1':
            hl = HullLoss(self.device)
            return hl.l1_norm(target_fol, target_lead, fol, lead)
            # xdiff = torch.sum(torch.abs(target_fol - fol), dim=1)
            # ydiff = torch.sum(torch.abs(target_lead - lead), dim=1)
        else:
            raise NotImplementedError()

    # =========== Preprocessing ==============

    def precompute_uniform_reach_probabilities(self, zero_on_terminal=False) -> List[float]:
        ret = [1.] * self.tree_game.num_infosets()
        for inf_id, inf in enumerate(self.tree_game.infosets):
            if inf.is_terminal() and zero_on_terminal:
                ret[inf_id] = 0.

            self_prob = ret[inf_id]
            if inf.player_to_move() == util.CHANCE:
                for child_idx, child_inf in enumerate(inf.child_states):
                    child_inf_id = child_inf.state_id
                    ret[child_inf_id] = self_prob * inf.probs[child_idx]
            else:
                num_child_states = len(inf.child_states)
                for child_idx, child_inf in enumerate(inf.child_states):
                    child_inf_id = child_inf.state_id
                    ret[child_inf_id] = self_prob / float(num_child_states)
        return ret

    # =========== Utility ==============

    def hull_to_list_of_tuples(self, hull_leader: util.Hull, hull_follower: util.Hull):
        list_of_tuples = []

        npoints = hull_leader.size()[0]
        for i in range(npoints):
            list_of_tuples.append(util.make_coord(
                hull_leader[i], hull_follower[i]))
        return list_of_tuples

    def list_of_tuples_to_hull(self, list_of_tuples):
        leader = torch.tensor([x[util.LEADER]
                               for x in list_of_tuples], device=self.device)
        follower = torch.tensor([x[util.FOLLOWER]
                                 for x in list_of_tuples], device=self.device)

        return util.Hull(follower, leader)

    def update_frozen_network(self) -> None:
        self.frozen_network = deepcopy(self.main_network)
        self.frozen_network_extractor.set_network(self.frozen_network)
