from policy.tree_policy import TreePolicy


class TabularTreePolicy(TreePolicy):
    def __init__(self, action_probs, promised_payoffs=None):
        self.action_probs = action_probs
        self.promised_payoffs = promised_payoffs

    def get_probs(self, inf_id):
        return self.action_probs[inf_id]

    def get_promised_payoffs(self, inf_id):
        if self.promised_payoffs is None:
            raise NotImplementedError()
        return self.promised_payoffs[inf_id]
