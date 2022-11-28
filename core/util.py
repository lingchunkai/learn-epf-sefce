import cProfile
import bisect
from bisect import bisect_left, bisect_right
from collections import namedtuple

PLAYER1 = 0
PLAYER2 = 1
CHANCE = 2

LEADER = PLAYER1
FOLLOWER = PLAYER2


def profileit(func):
    # TODO: credit the post for SO...
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile"  # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper


class KeyifyList(object):
    def __init__(self, inner, key):
        self.inner = inner
        self.key = key

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, k):
        return self.key(self.inner[k])


Hull = namedtuple('Hull', ['X', 'Y'])


def find_idx_ge(a, x, key=None, must_be_found=True, rightmost=False):
    if key is None:
        z = a
    else:
        z = KeyifyList(a, key)

    if not rightmost:
        'Find *index of* leftmost item greater than or equal to x'
        i = bisect_left(z, x)
    else:
        i = bisect_right(z, x)

    if i != len(a) or not must_be_found:
        return i
    raise ValueError


def lin_interpolate(x1, x2, y1, y2, target_x):
    # Compute linear interpolation at truncated point.
    x_ratio = (target_x - x1) / (x2-x1)
    y_diff = y2-y1
    return y_diff * x_ratio + y1


def make_coord(leader_val, follower_val):
    """ Converts leader and follower payoffs to a single coord,
        represented by a tuple.
    TODO: optimize.
    """
    ret = [None, None]
    ret[LEADER] = leader_val
    ret[FOLLOWER] = follower_val
    return tuple(ret)


def grad(p1, p2):
    """ Gradient of p2 wrt p1, where the x axis is given by follower payoff.
    """
    return (p2[LEADER] - p1[LEADER]) / (p2[FOLLOWER]-p1[FOLLOWER])
