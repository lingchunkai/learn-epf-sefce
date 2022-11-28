import torch
import torch.jit as jit


class HullLoss(object):
    """
    Compute different types of hull losses using mutual interpolation of 
    each hull. 
    NOTE: Each tensor have to be sorted such that for both mask1 and mask2,
        we have masked entries to the right (all the 1's are stuffed to the
        right.)
    TODO: Shift inside this function using this:
            X1, Y1, mask1 = self.ops.shift_masked_to_right(
                target_fol, target_lead, target_mask, [])
    """

    def __init__(self, device):
        self.device = device

    @jit.export
    def l2_norm(self, X1, Y1, X2, Y2, mask1, mask2):
        num_not_inf_1 = torch.sum((~mask1).long(), dim=1)
        num_not_inf_2 = torch.sum((~mask2).long(), dim=1)
        num_not_inf = num_not_inf_1+num_not_inf_2
        X_, Y1_, Y2_, mask = self.prep_norm(X1, Y1, X2, Y2, num_not_inf)

        return torch.sum((Y1_-Y2_)**2 * mask.long(), dim=1)

    @jit.export
    def l1_norm(self, X1, Y1, X2, Y2, num_not_inf=None):
        X_, Y1_, Y2_, mask = self.prep_norm(X1, Y1, X2, Y2, num_not_inf)
        return torch.sum(torch.abs(Y1_-Y2_) * mask.long(), dim=1)

    @jit.export
    def prep_norm(self, X1, Y1, X2, Y2, num_not_inf=None):

        tlen = X1.size()[1] + X2.size()[1]
        bsize = X1.size()[0]
        ints = torch.arange(0, tlen, device=self.device).unsqueeze(
            0).expand((bsize, -1))
        mask = ints < num_not_inf.unsqueeze(1).expand((-1, tlen))

        X_, Y1_, Y2_ = self.get_interp_fns(X1, Y1, X2, Y2)
        return X_, Y1_, Y2_, mask

    @jit.export
    def get_interp_fns(self, X1, Y1, X2, Y2):
        Y_QX1 = self.interp_points(X1, X2, Y2)
        Y_QX2 = self.interp_points(X2, X1, Y1)

        X_ = torch.concat([X1, X2], dim=1)
        Y1_ = torch.concat([Y1, Y_QX2], dim=1)
        Y2_ = torch.concat([Y_QX1, Y2], dim=1)

        X_, ind = torch.sort(X_, dim=1)
        Y1_ = torch.gather(Y1_, 1, ind)
        Y2_ = torch.gather(Y2_, 1, ind)

        return X_, Y1_, Y2_

    @jit.export
    def interp_points(self, qx, tx, ty):
        '''
        qx query x
        tx, ty target x,y
        Assumes that invalid points have already been pushed to the right side,
        the maximum x values in both query_x and target_x that are not INFINITY
        are the same (equal to the upper bound).
        '''

        ind = torch.searchsorted(tx, qx)

        # +1 and -1 are to make sure gradients are defined when extrapolated
        # to the right and left.
        # TODO: right interpolation could be wrong!!
        aug_tx = torch.concat([tx[:, 0:1]-1, tx, tx[:, -1:]+1], dim=1)
        aug_ty = torch.concat([ty[:, 0:1], ty, ty[:, -1:]], dim=1)

        left_x = torch.gather(aug_tx, 1, ind)
        right_x = torch.gather(aug_tx, 1, ind+1)
        left_y = torch.gather(aug_ty, 1, ind)
        right_y = torch.gather(aug_ty, 1, ind+1)

        p = (qx - left_x)/(right_x - left_x)

        ret_y = p * right_y + (1.-p) * left_y

        return ret_y


if __name__ == '__main__':

    """
    X1 = torch.tensor([[1., 3., 5., 7., 9.]])
    X2 = torch.tensor([[2., 4., 6., 8., 10.]])
    Y1 = torch.tensor([[1., 1., 1., 1., 1.]])
    Y2 = torch.tensor([[1., 1., 1., 1., 1.]]) * 3
    """

    X1 = torch.tensor([[1., 3., 5., 7.]])
    Y1 = torch.tensor([[1., 3., 5., 1., ]])
    X2 = torch.tensor([[2., 6., ]])
    Y2 = torch.tensor([[2., 2., ]])

    mask1 = torch.ones_like(X1) < 0
    mask2 = torch.ones_like(X2) < 0

    op = HullLoss(torch.device('cpu'))
    print(op.l2_norm(X1, Y1, X2, Y2, mask1, mask2))
