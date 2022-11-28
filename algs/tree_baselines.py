""" Fully cooperative and competitive outcomes.
"""

from core import util


def nash_subgame_perfect(tree_game):
    # Follower payoff
    f = [None] * len(tree_game.infosets)
    # leader payoff
    l = [None] * len(tree_game.infosets)

    for inf in reversed(tree_game.infosets):
        inf_id = inf.state_id
        if inf.is_terminal():
            f[inf_id] = inf.state.rewards()[util.FOLLOWER]
            l[inf_id] = inf.state.rewards()[util.LEADER]

        elif inf.state.player_to_move() == util.LEADER:
            # Take minimum of child states.
            if l[inf_id] is None:
                l[inf_id] = -float('inf')

            for child_state in inf.child_states:
                child_state_id = child_state.state_id
                if l[inf_id] < l[child_state_id]:
                    f[inf_id] = f[child_state_id]
                    l[inf_id] = l[child_state_id]

        elif inf.state.player_to_move() == util.FOLLOWER:
            # Take minimum of child states.
            if f[inf_id] is None:
                f[inf_id] = -float('inf')

            for child_state in inf.child_states:
                child_state_id = child_state.state_id
                if f[inf_id] < f[child_state_id]:
                    f[inf_id] = f[child_state_id]
                    l[inf_id] = l[child_state_id]

        elif inf.state.player_to_move() == util.CHANCE:
            # Weighted sum of children
            f[inf_id], l[inf_id] = 0.0, 0.0
            for idx, child_state in enumerate(inf.child_states):
                child_state_id = child_state.state_id
                prob = inf.probs[idx]
                f[inf_id] += prob * f[child_state_id]
                l[inf_id] += prob * l[child_state_id]

    return f, l


def cooperative(tree_game):
    """ Solve general sum game as if it were fully cooperative, 
        from the perspective of the follower

    returns (follower payoff, leader payoff)
    """

    # Follower payoff
    v = [None] * len(tree_game.infosets)
    # leader payoff
    l = [None] * len(tree_game.infosets)

    for inf in reversed(tree_game.infosets):
        inf_id = inf.state_id
        if inf.is_terminal():
            v[inf_id] = inf.state.rewards()[util.FOLLOWER]
            l[inf_id] = inf.state.rewards()[util.LEADER]

        elif inf.state.player_to_move() == util.LEADER or \
                inf.state.player_to_move() == util.FOLLOWER:
            # Take minimum of child states.
            if v[inf_id] is None:
                v[inf_id] = -float('inf')

            for child_state in inf.child_states:
                child_state_id = child_state.state_id
                if v[inf_id] < v[child_state_id]:
                    v[inf_id] = v[child_state_id]
                    l[inf_id] = l[child_state_id]

        elif inf.state.player_to_move() == util.CHANCE:
            # Weighted sum of children
            v[inf_id], l[inf_id] = 0.0, 0.0
            for idx, child_state in enumerate(inf.child_states):
                child_state_id = child_state.state_id
                prob = inf.probs[idx]
                v[inf_id] += prob * v[child_state_id]
                l[inf_id] += prob * l[child_state_id]

    return v, l


def competitive(tree_game):
    """ Solve zero-sum game where the leader takes
        plays to hurt the follower as much as possible.
    """
    # Contains follower payoffs.
    v = [None] * len(tree_game.infosets)
    l = [None] * len(tree_game.infosets)
    threats = [None] * len(tree_game.infosets)

    for inf in reversed(tree_game.infosets):
        inf_id = inf.state_id
        if inf.is_terminal():
            v[inf_id] = inf.state.rewards()[util.FOLLOWER]
            l[inf_id] = inf.state.rewards()[util.LEADER]
        elif inf.state.player_to_move() == util.LEADER:
            # Take minimum of child states.
            if v[inf_id] is None:
                v[inf_id] = float('inf')

            for child_state in inf.child_states:
                child_state_id = child_state.state_id
                if v[inf_id] > v[child_state_id]:
                    v[inf_id] = v[child_state_id]
                    l[inf_id] = l[child_state_id]

        elif inf.state.player_to_move() == util.FOLLOWER:
            # Take maximum of child states and take value of threats.
            if v[inf_id] is None:
                v[inf_id] = -float('inf')

            threat_vals = []  # Contains threats
            for child_state in inf.child_states:
                child_state_id = child_state.state_id
                if v[inf_id] < v[child_state_id]:
                    v[inf_id] = v[child_state_id]
                    l[inf_id] = l[child_state_id]

                # Compute value of threats
                threat_vals.append((v[child_state_id], child_state_id))
            threat_vals.sort()

            # If there is only one action for the follower,
            # then there is no threat required, follower only has one option.
            if len(inf.child_states) == 1:
                threats[inf.child_states[0].state_id] = -float('inf')
                continue

            # Set weakest threat to be highest of second highest.
            for child_state in inf.child_states:
                child_state_id = child_state.state_id
                if child_state_id != threat_vals[-1][1]:
                    # Use the weakest threat value (highest follower payoff)
                    threats[child_state_id] = threat_vals[-1][0]
                    # threats[child_state_id] = -float('inf')
                else:
                    # The child is the weakest threat, so the threat can be the second highest.
                    threats[child_state_id] = threat_vals[-2][0]
                    # threats[child_state_id] = -float('inf')
        elif inf.state.player_to_move() == util.CHANCE:
            # Weighted sum of children
            v[inf_id], l[inf_id] = 0.0, 0.0
            for idx, child_state in enumerate(inf.child_states):
                child_state_id = child_state.state_id
                prob = inf.probs[idx]
                v[inf_id] += prob * v[child_state_id]
                l[inf_id] += prob * l[child_state_id]

    return (v, l), threats
