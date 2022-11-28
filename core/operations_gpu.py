import torch
import torch.jit as jit

import numpy as np
import matplotlib.pyplot as plt


class OperationsGPU(object):
    """ TODO:
        1) Create composite operations for .cost_matrix() and .points_not_in_envelope(),
        reuse tensors where possible.
        2) Change truncate to accept mask.
    """

    def __init__(self, device):
        self.device = device

    @jit.export
    def upper_concave_envelope(self, X, Y, mask=None):
        """ Gives the concave sorted version of X, Y, assuming they may
        be masked already. Note: we internally perform sorting here.
        TODO: allow option to avoid sorting if input is already sorted.

        returns: X, Y points which have been sorted in X, and
        indices_to_remove: a boolean tensor of size like X which 
            containing True where things *need* to be removed.
        """
        sorted_X, indices = torch.sort(X, dim=1)
        sorted_Y = torch.gather(Y, 1, indices)

        tpoints = sorted_X.shape[1]

        if mask is None:
            mask = torch.ones_like(X) < 0

        indices_to_remove = self.points_not_in_envelope(
            sorted_X, sorted_Y, tpoints, mask)

        return sorted_X, sorted_Y, indices_to_remove

    @jit.export
    def points_not_in_envelope(self,
                               sorted_X,
                               sorted_Y,
                               tpoints,
                               mask):
        """ Get mask of points not in envelope (i.e., within the upper concave
        envelope)

        NOTE: this does very expensive cubic time (in the number of points in the hull)
            computation. How to avoid?
        """
        X1 = sorted_X.unsqueeze(2).unsqueeze(
            3).expand((-1, -1, tpoints, tpoints))
        X2 = sorted_X.unsqueeze(1).unsqueeze(
            3).expand((-1, tpoints, -1, tpoints))
        Y1 = sorted_Y.unsqueeze(2).unsqueeze(
            3).expand((-1, -1, tpoints, tpoints))
        Y2 = sorted_Y.unsqueeze(1).unsqueeze(
            3).expand((-1, tpoints, -1, tpoints))

        X3 = sorted_X.unsqueeze(1).unsqueeze(
            2).expand((-1, tpoints, tpoints, -1))
        Y3 = sorted_Y.unsqueeze(1).unsqueeze(
            2).expand((-1, tpoints, tpoints, -1))

        m1 = mask.unsqueeze(2).unsqueeze(3).expand((-1, -1, tpoints, tpoints))
        m2 = mask.unsqueeze(1).unsqueeze(3).expand((-1, tpoints, -1, tpoints))
        masked_pair = torch.logical_or(m1, m2)

        # Check if point lies above or below
        cross = (X2-X1) * (Y2-Y3) - (Y2-Y1) * (X2-X3) >= 0

        # The pair of points responsible for this point to be omitted
        # must both not be masked
        cross = torch.logical_and(cross, ~masked_pair)

        bounds_lo = X1 - X3 < 0
        bounds_hi = X2 - X3 > 0

        q = torch.logical_and(torch.logical_and(bounds_lo, bounds_hi), cross)
        to_remove = q.any(dim=1).any(dim=1)

        return to_remove

    @jit.export
    def mask_of_repetitions(self, X, Y, mask):
        """
        Mask out any repetitions, take the larger one.
        NOTE: this is pretty slow, does pairwise comparisons....
        """
        tpoints = X.shape[1]
        X1 = X.unsqueeze(2).expand((-1, -1, tpoints))
        X2 = X.unsqueeze(1).expand((-1, tpoints, -1))
        Y1 = Y.unsqueeze(2).expand((-1, -1, tpoints))
        Y2 = Y.unsqueeze(1).expand((-1, tpoints, -1))

        log = torch.logical_and(X1-X2 == 0, Y1 < Y2)
        # The element Y2 that is higher must not be masked!
        log = torch.logical_and(
            log, ~(mask.unsqueeze(1).expand((-1, tpoints, -1))))
        return torch.any(log, dim=2)

    @jit.export
    def truncate(self, X, Y, cutoff, mask):
        """ Truncates (and interpolates) all points below cutoff.
            Masks those entries that are strictly lower than cutoff and 
            replaces one (if any) of them by the interpolated value.

            Returns: new_mask (which includes effect of old mask)
        """
        X_, Y_ = X, Y

        # Get points we will not include (directly) in the the truncated points.
        to_cut = X < cutoff.unsqueeze(1).expand((-1, X.shape[1]))
        new_mask = torch.logical_or(to_cut, mask)

        # Get the first point which we *are* including
        idx_first_higher_eq = torch.argmin(to_cut.long(), dim=1)
        y_right = torch.gather(Y, 1, idx_first_higher_eq.unsqueeze(1))
        x_right = torch.gather(X, 1, idx_first_higher_eq.unsqueeze(1))
        # TODO: assert new_mask is not all True, since there must be at least one point remaining

        # Now, get previous index that was cut but *not* masked.
        # Remember this could be empty!
        # Get points which cannot be the interpolated point
        mask_tmp = X >= cutoff.unsqueeze(1).expand(
            (-1, X.shape[1]))  # Note the equality
        mask_tmp = torch.logical_or(mask_tmp, mask)
        # Do trick using arange to get highest index.
        idx_tmp = torch.arange(0, X.shape[1], device=self.device).unsqueeze(
            0).expand((X.shape[0], -1))
        idx_tmp = idx_tmp - mask_tmp.long() * (1 + X.shape[1])
        pinpoint = x_right.squeeze(1) == cutoff
        need_to_interpolate = torch.any(~mask_tmp, dim=1)
        need_to_interpolate = torch.logical_and(need_to_interpolate, ~pinpoint)
        need_to_interpolate = torch.logical_and(
            need_to_interpolate, ~torch.all(to_cut, dim=1))
        # Contains indices to interpolate with (for those rows that need to be)
        idx_to_interp = torch.argmax(idx_tmp, dim=1)
        # Number of rows which we need to do interpolating
        num_interpolate = torch.sum(need_to_interpolate.long())

        # Now, we have 2 cases.
        # case 1: need_to_interpolate[row] is False, no point to interpolate --- done
        # case 2: need_to_interpolate[row] is True, interpolate with that point

        # We compute the gradients and get interpolated point.
        y_left = torch.gather(Y, 1, idx_to_interp.unsqueeze(1))
        x_left = torch.gather(X, 1, idx_to_interp.unsqueeze(1))

        grad_extra = (y_right - y_left)/(x_right-x_left)
        y_midway = y_left + grad_extra * \
            (cutoff.unsqueeze(1) - x_left)

        # For rows which need to have a new point added,
        # we select an index (the the point to the left in the highest index
        # is definitely masked (and not at index 0), so we just replace
        # it by the new point.), and replace them with the new point values
        # TODO: speed this logic up.
        X_[need_to_interpolate, :] = X_[need_to_interpolate, :].scatter(
            1,
            (idx_first_higher_eq[need_to_interpolate] - 1).unsqueeze(1),
            cutoff[need_to_interpolate].unsqueeze(1),
        )
        Y_[need_to_interpolate, :] = Y_[need_to_interpolate, :].scatter(
            1,
            (idx_first_higher_eq[need_to_interpolate] - 1).unsqueeze(1),
            y_midway[need_to_interpolate])

        # Reset all the new points to not be masked.
        new_mask[need_to_interpolate, :] = new_mask[need_to_interpolate, :].scatter(1,
                                                                                    (idx_first_higher_eq[need_to_interpolate] - 1).unsqueeze(
                                                                                        1),
                                                                                    torch.ones(num_interpolate, device=self.device).unsqueeze(1) < 0)  # False

        return X_, Y_, new_mask

    @jit.export
    def shift_masked_to_right(self, X, Y, mask, list_tags):
        """
            Shift all entries such that masked elements are to the right.
            Note we modify the x locations for this; we do not set them
            to +inf because that messes up future elements, but we set them 
            to be 1 + the maximum of the entries.

        list_tags: list of tensors shaped like X which we will sort accordingly.
        """
        X_, Y_ = X, Y
        maxx = torch.max(X, dim=1)[0] + 1
        X_[mask] = maxx.unsqueeze(1).expand((-1, X.size()[1]))[mask]
        X_, indices = torch.sort(X, dim=1)
        Y_ = torch.gather(Y_, 1, indices)
        new_mask = torch.gather(mask, 1, indices)

        ret_tags = []
        for tag in list_tags:
            ret_tags.append(torch.gather(tag, 1, indices))

        return X_, Y_, new_mask, *tuple(ret_tags)

    @jit.export
    def extract_decreasing(self, X, Y, mask):
        """ 
            Sets entries that are below and to the left of the maximum Y value to 
            be masked. So, if X is sorted and concave, then we get the frontier.
        """
        Y_ = Y - 2 * mask.long() * (torch.max(X, dim=1)
                                    [0] - torch.min(X, dim=1)[0]).unsqueeze(1).expand_as(X)
        idx = torch.argmax(Y_, dim=1)

        z = torch.arange(0, X.size()[1], device=self.device).unsqueeze(
            0).expand_as(X)
        mask_ = z < idx.unsqueeze(1).expand_as(X)

        return mask_

    @jit.export
    def truncate_ub(self,
                    X_: torch.Tensor,
                    Y_: torch.Tensor,
                    mask: torch.Tensor,
                    ub: torch.Tensor,
                    eps=1e-4):

        # TODO: test

        interp_left_exists = X_ <= ub.unsqueeze(1).expand_as(X_)
        interp_left_exists = torch.logical_and(interp_left_exists, ~mask)
        interp_left = (~interp_left_exists).long() * - \
            (interp_left_exists.size()[1] * 2)
        rnge = torch.arange(0, interp_left.size()[1]).to(self.device)
        rnge = rnge.unsqueeze(0).expand((X_.size()[0], -1))
        left_idx = torch.argmax(rnge + interp_left, dim=1)

        interp_right_exists = X_ > ub.unsqueeze(1).expand_as(X_)
        interp_right_exists = torch.logical_and(
            interp_right_exists, ~mask)
        interp_right = (~interp_right_exists).long() * \
            interp_right_exists.size()[1] * 2
        right_idx = torch.argmin(rnge + interp_right, dim=1)

        need_to_interp = torch.logical_and(
            torch.any(interp_left_exists, dim=1),
            torch.any(interp_right_exists, dim=1)
        )
        need_to_collapse = torch.all(~interp_left_exists, dim=1)

        # Interpolation
        x_left = torch.gather(X_, 1, left_idx.unsqueeze(1))
        y_left = torch.gather(Y_, 1, left_idx.unsqueeze(1))

        x_right = torch.gather(X_, 1, right_idx.unsqueeze(1))
        y_right = torch.gather(Y_, 1, right_idx.unsqueeze(1))

        grad_extra = (y_right - y_left)/(x_right-x_left)
        y_midway = y_left + grad_extra * \
            (ub.unsqueeze(1) - x_left)

        X_[need_to_interp, :] = X_[need_to_interp, :].scatter(
            1,
            right_idx[need_to_interp].unsqueeze(1),
            ub[need_to_interp].unsqueeze(1)
        )

        Y_[need_to_interp, :] = Y_[need_to_interp, :].scatter(
            1,
            right_idx[need_to_interp].unsqueeze(1),
            y_midway[need_to_interp]
        )

        new_mask = torch.logical_or(
            mask, X_ > (ub).unsqueeze(1).expand_as(X_))
        new_mask[need_to_interp, :] = new_mask[need_to_interp, :].scatter(
            1,
            right_idx[need_to_interp].unsqueeze(1),
            torch.ones((X_.size()[0], 1), device=self.device)[
                need_to_interp] < 0
        )

        # collapse
        X_[need_to_collapse, :] = X_[need_to_collapse, :].scatter(
            1,
            right_idx[need_to_collapse].unsqueeze(1),
            ub[need_to_collapse].unsqueeze(1)
        )
        new_mask[need_to_collapse, :] = new_mask[need_to_collapse, :].scatter(
            1,
            right_idx[need_to_collapse].unsqueeze(1),
            torch.ones((X_.size()[0], 1), device=self.device)[
                need_to_collapse] < 0
        )

        return X_, Y_, new_mask
