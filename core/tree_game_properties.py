import torch
from core.perfect_game import TreeState, TreeGame
from algs.tree_baselines import competitive, cooperative
from typing import List, Tuple


class TreeGameProperties(object):
    def __init__(self, tree_game: TreeGame, featurizer):
        self.tree_game = tree_game
        self.featurizer = featurizer

    def preprocess(self):
        self.feats = self.precompute_all_features()
        (self.f_compete, self.l_compete), self.threats = competitive(self.tree_game)
        self.f_coop, self.l_coop = cooperative(self.tree_game)
        self.inf_ids_non_term = self.preprocess_nonterminal_inf_ids()
        self.branching_factor = self.preprocess_branching_factor()

    def precompute_all_features(self) -> List[torch.tensor]:
        """ Precompute features of all reachable states.
        """
        feats = []

        for inf_id, inf in enumerate(self.tree_game.infosets):
            feat = self.featurizer(inf.state)
            feats.append(feat)
        return feats

    def preprocess_nonterminal_inf_ids(self) -> List[int]:
        # Get all non-terminal states
        inf_ids_non_term = []
        for inf_id, inf in enumerate(self.tree_game.infosets):
            if inf.is_terminal():
                continue
            inf_ids_non_term.append(inf_id)
        return inf_ids_non_term

    def preprocess_branching_factor(self):
        return max([len(inf.child_states) for inf in self.tree_game.infosets])
