import sys
sys.path.append('core')

import os

import numpy as np
import torch
import torch.nn.functional as F

from .RAFT.core.raft import RAFT
from .RAFT.core.utils.utils import InputPadder

DEVICE = 'cuda'


class ConsistencyMetric:

    def __init__(self):
        super(ConsistencyMetric, self).__init__()

        class Object(dict):
            pass

        args = Object()
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = True

        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'metrics', 'raft-things.pth')))

        self.model = model.module
        self.model.to(DEVICE)
        self.model.eval()

    def _warp(self, x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid - flo
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid)
        mask = torch.ones(x.size()).to(DEVICE)
        mask = F.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        return output, mask

    def compute(self, first_frame, second_frame):
        first_frame = (first_frame * 255.0).to(DEVICE)
        second_frame = (second_frame * 255.0).to(DEVICE)

        padder = InputPadder(first_frame.shape)
        first_frame, second_frame = padder.pad(first_frame, second_frame)

        _, flow_up = self.model(first_frame, second_frame, iters=20, test_mode=True)

        warped, mask = self._warp(first_frame, flow_up)
        second_frame = second_frame[0].permute(1, 2, 0).cpu().numpy()
        warped = warped[0].permute(1, 2, 0).cpu().numpy()

        mask = mask.squeeze().permute(1, 2, 0).cpu().numpy()
        non_occluded_pixel_count = np.count_nonzero(mask == 1) / 3.0
        l2_norm = np.linalg.norm(second_frame - warped)

        return torch.tensor((1.0 / non_occluded_pixel_count) * l2_norm)
