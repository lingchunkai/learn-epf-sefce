from __future__ import annotations
from math import trunc
from core import util
from itertools import groupby
from copy import deepcopy
from typing import Iterable, List, Tuple, Any
import numba
import numpy as np


def concave_envelope(points_list: List[np.ndarray],
                     redundant_eps: float = 0.0
                     ) -> Tuple[List(Tuple[float, float]), List[int]]:

    # Remove elements in point_list which are empty. Also construct
    # an index-to-action-index array used to map back to
    # action indices.
    points_list_remove_empty = []
    act_ids = []
    for i in range(len(points_list)):
        if points_list[i].shape[0] > 0:
            # Extract all nonempty hulls.
            # Note: cast here to float32 to make sure all arrays are of same type!
            # TODO: make sure all code calling this has points_list to be of dtype=np.float32.
            points_list_remove_empty.append(
                np.array(points_list[i], dtype=np.float32))
        for j in range(points_list[i].shape[0]):
            act_ids.append(i)
    # Action ids have to be cast to a numpy array for JIT to work.
    act_ids = np.array(act_ids)

    # Special case when *all* elements in points_list are empty (this shouldn't happen in practice).
    # We return a degenerate tuple of 0's then. Note that trying to return this in the JIT
    # function can result in errors.
    if len(points_list_remove_empty) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    # points_list_remove_empty = tuple(points_list_remove_empty)
    cat = np.concatenate(points_list_remove_empty, axis=0)
    ret = concave_envelope_(cat, act_ids, redundant_eps)
    return ret


@numba.jit(nopython=True,
           locals={'eps': numba.types.float32})
def concave_envelope_(cat,
                      act_ids,
                      redundant_eps: float = 0.0,
                      ) -> Tuple[List(Tuple[float, float]), List[int]]:
    """ Vectorized version of concave envelope that is supposed to be compatible with numba.
    """

    # Tolerance to lock in a new point, tol for x1, x2 to be considered
    # the "same" point.
    eps = 1e-6

    # Sort giant array.
    # lexsort priority decreases from rightmost to leftmost column.
    # sort_ind = np.lexsort(cat) # lexsort is not supported yet.
    indices = np.argsort(cat[:, -1])

    trimmed_ids = [0 for x in range(0)]
    x = cat[indices[0], util.FOLLOWER]
    best_y = cat[indices[0], util.LEADER]
    best_h = 0
    total_points = cat.shape[0]
    for h in range(1, total_points + 1):
        if h == total_points:
            trimmed_ids.append(indices[best_h])
            break
        if cat[indices[h], util.FOLLOWER] > x + eps:
            # lock in new line
            trimmed_ids.append(indices[best_h])
            if h < total_points:
                # start new line
                best_y = cat[indices[h], util.LEADER]
                x = cat[indices[h], util.FOLLOWER]
                best_h = h
        else:
            # if new x value lies in [x, eps], let this be the higher value.
            x = cat[indices[h], util.FOLLOWER]
            if best_y < cat[indices[h], util.LEADER]:
                best_y = cat[indices[h], util.LEADER]
                best_h = h

    trimmed_acts = [act_ids[x] for x in trimmed_ids]
    trimmed_ids_array = np.array(trimmed_ids)
    trimmed_values = cat[trimmed_ids_array, :]

    # Compute indices of envelope based on trimmed indices.
    included_indices_of_giant = compute_indices_of_envelope_(trimmed_values)
    included_indices_of_giant_array = np.array(included_indices_of_giant)

    envelope_points = trimmed_values[included_indices_of_giant_array, :]
    envelope_tags = [trimmed_acts[z] for z in included_indices_of_giant]
    # TODO: make JIT compatible: Remove redundant points if specfied.
    '''
    if redundant_eps > 0.:
        envelope_points, envelope_tags = remove_redundant_points(
            envelope_points, redundant_eps, envelope_tags)
    '''
    return envelope_points, envelope_tags


def truncate(coords: np.ndarray, trunc_pos: float) -> List[Tuple[float, float]]:
    """ Only admit points that that have have follower payoff greater than trunc_pos
        Assumes that no repeated follower payoffs exist.
    """
    eps = 1e-4

    if coords.shape[0] == 0:
        return np.zeros((0, 2))

    # If truncation is too high, nothing is contained.
    if trunc_pos > coords[-1, util.FOLLOWER]+eps:
        return np.zeros((0, 2))

    # If truncation is too low, everything is contained
    if trunc_pos < coords[0, util.FOLLOWER]:
        return deepcopy(coords)

    # If truncation is precisely on the highest follower payoff (up to epsilon),
    # then use that last point's *value* but with the truncated x value
    if trunc_pos >= coords[-1, util.FOLLOWER]-eps:
        return np.array([[coords[-1, 0], trunc_pos]])

    # Now, we are sure *some line segment* is truncated. We iterate through line segments
    # and find the first intersection found.
    new_point = None
    for idx in range(1, len(coords)):
        if trunc_pos < coords[idx][util.FOLLOWER]:
            # Contains replacement for point which was cut off.
            new_point = [None, None]
            new_point[util.FOLLOWER] = trunc_pos

            new_leader_util = util.lin_interpolate(coords[idx-1, util.FOLLOWER],
                                                   coords[idx, util.FOLLOWER],
                                                   coords[idx - 1,
                                                          util.LEADER],
                                                   coords[idx, util.LEADER],
                                                   trunc_pos)
            new_point[util.LEADER] = new_leader_util
            break

    assert new_point is not None
    leftover_points = np.concatenate(
        [
            np.array([new_point]),
            coords[idx:, :]
        ],
        axis=0
    )

    return leftover_points


def extract_maximum_idx(hull):
    # Extracts decreasing portion of value function.
    # Asssmes that hull is concave.

    # Find maximum y value and start from there
    for idx in range(hull.shape[0]):
        if idx == hull.shape[0] - 1:
            return idx
        if hull[idx, util.LEADER] > hull[idx+1, util.LEADER]:
            return idx


def concave_envelope_old(points_list: List[List[Tuple[float, float]]],
                         redundant_eps: float = 0.0
                         ) -> Tuple[List(Tuple[float, float]), List[int]]:
    """
    points_list: list of lists of(., .) points, each point containing
        leader and follower payoffs.
    redundant_eps: When the concave envelope has 3 or more colinear points
                   we can try to remove them. `redundant_eps` specifies the
                   tolerance for detecting colinearity.

                   Note: the default of value 0.0 will mean that no trimming is done.

    returns tuple of
        (i) list of tuples(points) forming the upper concave envelope,
        (ii) action ids for each of the points.

    Note: Input points need not be concave themselves, since internally we are
          concatenating everything to one giant set.
    """
    if len(points_list) == 0:
        return [], []

    # Concatenate all points into a giant list of points.
    # Include which block was in before concatanation, since we need to return
    # which action we are interpolating over.
    points_cat = [(*p, id)
                  for (id, points) in enumerate(points_list) for p in points]

    # Sort based on follower payoff.
    points_cat.sort(key=lambda x: x[util.FOLLOWER])

    # Remove duplicates in follower payoff (duplicates could occur over
    # different actions), take max of leader payoff when there are duplicates.
    groups = groupby(points_cat, key=lambda x: x[util.FOLLOWER])
    points_cat = [max(g, key=lambda x: x[util.LEADER]) for k, g in groups]

    # Unzip coordinates (points) and action ids (tags).
    points = [(x[0], x[1]) for x in points_cat]
    tags = [x[-1] for x in points_cat]

    # Run through the beneath beyond algorithm to figure out which
    # points are included in the final envelope.
    included_idx = compute_indices_of_envelope(points)
    # included_idx = scipy_compute_indices_of_envelope(points)
    envelope_points = [points[idx] for idx in included_idx]
    envelope_tags = [tags[idx] for idx in included_idx]

    # Remove redundant points if specfied.
    if redundant_eps > 0.:
        envelope_points, envelope_tags = remove_redundant_points(
            envelope_points, redundant_eps, envelope_tags)

    return envelope_points, envelope_tags


@numba.jit(nopython=True)
def compute_indices_of_envelope_(points):
    incl_idx = [0]
    incl_grad_num = []
    incl_grad_denom = []
    for idx in range(1, len(points)):
        # Keep removing previously added points
        # until the gradient is non-increasing.
        while(True):
            # Gradient formed from this new point (idx)
            # and the last point currently added to the convex
            # hull.
            # If this gradient is less than the previous gradient,
            # then we need to remove the last point in our current
            # proposed hull.
            new_grad_num = points[idx, 0] - points[incl_idx[-1], 0]
            new_grad_denom = points[idx, 1] - points[incl_idx[-1], 1]

            # Check if there are still line segments to be removed.
            if len(incl_grad_num) > 0:
                # Perform the comparison "new_grad >= incl_grad[-1]" is computed
                # by using num_1 * dem_2 >= (<=) num_2 * dem1, where the sign
                # is obtain by ss, the product of denominators.
                ss = new_grad_denom * incl_grad_denom[-1]
                q1 = new_grad_num * incl_grad_denom[-1]
                q2 = incl_grad_num[-1] * new_grad_denom
                if (ss > 0. and q1 >= q2) or (ss < 0. and q1 < q2):
                    incl_idx.pop()
                    incl_grad_num.pop()
                    incl_grad_denom.pop()
                else:
                    break
            else:
                # There are no possible line segments to be removed.
                break

        # Add this point and its gradient from the point it is connected to,
        # assuming it remains in the convex hull.
        incl_idx.append(idx)
        incl_grad_num.append(new_grad_num)
        incl_grad_denom.append(new_grad_denom)

    return incl_idx


@numba.jit(nopython=True)
def jit_grad(p1, p2):
    """ Gradient of p2 wrt p1, where the x axis is given by follower payoff.
    """
    return (p2[0] - p1[0]) / (p2[1]-p1[1])


def compute_indices_of_envelope(points: List[Tuple[float, float]]) -> List[int]:
    """ Returns indices of points which are are included in the upper concave
        envelope of `points`.
    points: list of coordinates sorted by increasing follower payoffs.

    Method:
        Beneath-beyond method.
        We keep adding points ensuring gradient was decreasing. If new point
        has a gradient that increases, we need to 'backtrack' and remove previously
        added points until the new gradient to be added is less than the previous gradient.
    """
    if len(points) == 0:
        return []

    incl_idx = [0]
    incl_grad = [float('inf')]

    for idx in range(1, len(points)):
        # Keep removing previously added points
        # until the gradient is non-increasing.
        while(True):
            # Gradient formed from this new point (idx)
            # and the last point currently added to the convex
            # hull.
            new_grad = util.grad(
                points[incl_idx[-1]], points[idx])
            # If this gradient is less than the previous gradient,
            # then we need to remove the last point in our current
            # proposed hull.
            if new_grad >= incl_grad[-1]:
                incl_idx.pop()
                incl_grad.pop()
            else:
                break

        # Add this point and its induced gradient.
        incl_idx.append(idx)
        incl_grad.append(new_grad)

    return incl_idx


def truncate_old(coords: List[Tuple[float, float]], trunc_pos: float) -> List[Tuple[float, float]]:
    """ Only admit points that that have have follower payoff greater than trunc_pos
        Assumes that no repeated follower payoffs exist.
    """
    if len(coords) == 0:
        return []

    # If truncation is too high, nothing is contained.
    if trunc_pos > coords[-1][util.FOLLOWER]:
        return []

    # If truncation is too low, everything is contained
    if trunc_pos < coords[0][util.FOLLOWER]:
        return deepcopy(coords)

    # If truncation is precisely on the highest follower payoff,
    # then use that point.
    if trunc_pos == coords[-1][util.FOLLOWER]:
        return [coords[-1]]

    # Now, we are sure *some line segment* is truncated. We iterate through line segments
    # and find the first intersection found.
    new_point = None
    for idx in range(1, len(coords)):
        if trunc_pos < coords[idx][util.FOLLOWER]:
            # Contains replacement for point which was cut off.
            new_point = [None, None]
            new_point[util.FOLLOWER] = trunc_pos

            new_leader_util = util.lin_interpolate(coords[idx-1][util.FOLLOWER],
                                                   coords[idx][util.FOLLOWER],
                                                   coords[idx -
                                                          1][util.LEADER],
                                                   coords[idx][util.LEADER],
                                                   trunc_pos)
            new_point[util.LEADER] = new_leader_util
            break

    assert new_point is not None
    leftover_points = [tuple(new_point)]
    for i in range(idx, len(coords)):
        leftover_points.append(coords[i])

    return leftover_points


def remove_repeats(coords: List[Tuple[float, float]]):
    """ Remove all entries where both coordindates are identical.
    """
    ret = []
    for c in coords:
        if len(ret) > 0 and c[util.FOLLOWER] == ret[-1][util.FOLLOWER]:
            continue
        ret.append(c)
    return ret


def minkowski(coords_: List[List[Tuple[float, float]]],
              scales: List[float] = None,
              redundant_eps: float = 0.0) -> Tuple[List[Tuple[float, float]], List[int]]:
    """ Finds minkowski sum over list of list of coords

        cords: list of size n, each a list containing (sorted based
        on folower payoffs) coordinates of hulls.

        scale: list of size n, each containing a scalar (possibly negative,
               but typically a probability vector, of weight for each of
               the n lists. Defaults to a vector of 1's.)

        redundant_eps: tolerance for removing colinear points, 0.0 to not remove.
    """
    if scales is None:
        scales = [1.0] * len(coords_)

    # Remove duplicates in follower payoff. This can happen during
    # degenerate cases.
    coords = []
    for l in coords_:
        coords.append(remove_repeats(l))

    leftmost_fol = sum([c[0][util.FOLLOWER] * scales[i]
                        for i, c in enumerate(coords)])
    leftmost_lead = sum([c[0][util.LEADER] * scales[i]
                         for i, c in enumerate(coords)])

    # Assume that player actions are already sorted in increasing order.
    segments = []
    for action_idx, c in enumerate(coords):
        seg = construct_line_segments(c, scales[action_idx])
        for s in seg:
            segments.append((*s, action_idx))

    segments.sort(reverse=True)

    right_end_hull_actions = [None]
    coords_ret = [util.make_coord(leftmost_lead, leftmost_fol)]

    for grad, length, owner in segments:
        new_leader = grad * length + coords_ret[-1][util.LEADER]
        new_follower = length + coords_ret[-1][util.FOLLOWER]
        right_end_hull_actions.append(owner)
        coords_ret.append(util.make_coord(new_leader, new_follower))

    if redundant_eps > 0.:
        coords_ret, right_end_hull_actions = remove_redundant_points(
            coords_ret, redundant_eps, right_end_hull_actions)

    return coords_ret, right_end_hull_actions


def remove_redundant_points(coords: List(Tuple[float, float]), eps: float, *argv: List[List[Any]]):
    """ Removes inernal (non-end) points which approximately lie in the center of another line segment. These are
        the "useless" points.

        coords: list of 2-tuples
        eps: tolerance for y-deviations for each internal point (x_i, y_i) with respect to the straight line
             formed by (x_{i-1}, y_{i-1}) and (x_{i+1}, x_{i+1}).
        argv: list of iterables, each of the same size of coords.

        return 1 + len(argv) lists, containing only entries which are not colinear.

        Note about tolerances
        =====================
        In practice, there can be multiple points lying (approximately) on the same straight line.

        Consider the case where points 4 and 5 both lie on the line segment formed by points 3 and 6.
        Since errors would accumulate, if we remove both points 4, 5, then the tolerance for them could potentially
        exceed tol. However, the effect is at most linear in the number of points, which we assume to be not
        too large.
    """

    indices = remove_redundant_points_(coords, eps)

    # Note that non_colinear_idx are using indices from the already-processed
    # indices of `envelope_points`.
    new_points = [coords[idx] for idx in indices]

    if len(argv) > 0:
        ret_others = []
        for others in argv:
            ret_others.append([others[idx] for idx in indices])

        return new_points, *ret_others
    else:
        return new_points


def remove_redundant_points_(coords: List(Tuple[float, float]), eps: float, ) -> List[int]:
    assert eps > 0., 'Redandant points tolerance must be strictly > 0'

    idx_included = [True] * len(coords)
    for idx in range(1, len(coords)-1):
        # Gradient for line segmented formed by left and right points.
        interp_y = util.lin_interpolate(coords[idx-1][util.LEADER],
                                        coords[idx+1][util.LEADER],
                                        coords[idx-1][util.FOLLOWER],
                                        coords[idx + 1][util.FOLLOWER],
                                        coords[idx][util.LEADER])
        if abs(interp_y - coords[idx][util.FOLLOWER]) < eps:
            idx_included[idx] = False

    indices = [i for i in range(len(coords)) if idx_included[i]]

    return indices


def construct_line_segments(coords: List[Tuple[float, float]], scale: List[int]) -> List[Tuple[float, float]]:
    """ Extract gradients and lengths of line segments
        from a list of coords.
        If coords has size m (m points), then this returns a list of tuples
        of size m-1 with elements containing (grad, length)

        coords: list of 2-tuples
        scale: scalar containing scaling factor.
    """
    ret = []
    for i in range(1, len(coords)):
        length = (coords[i][util.FOLLOWER] -
                  coords[i-1][util.FOLLOWER]) * scale
        gradient = util.grad(coords[i-1], coords[i])
        ret.append((gradient, length))

    return ret
