from loss.utils import *
from loss.losses import *
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


class lossBuilder(nn.Module):
    def __init__(self) -> None:
        super(lossBuilder, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.crit_rot = BinRotLoss()

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        return output

    def forward(self, output, batch):
        heads = {'hm', 'dim', 'dep', 'rot'}
        losses = {head: 0 for head in heads}  # 对每一个head有对应的loss

        output = self._sigmoid_output(output)

        if 'hm' in output:
            losses['hm'] += checkpoint(self.crit, output['hm'], batch['hm'], batch['ind'],
                                       batch['mask'], batch['cat'])
            # losses['hm'] += self.crit(
            #     output['hm'], batch['hm'], batch['ind'],
            #     batch['mask'], batch['cat'])

        regression_heads = ['dep', 'dim']

        for head in regression_heads:
            if head in output:
                losses[head] += checkpoint(self.crit_reg, output[head], batch[head + '_mask'],
                                           batch['ind'], batch[head])
                # losses[head] += self.crit_reg(
                #     output[head], batch[head + '_mask'],
                #     batch['ind'], batch[head])

        if 'rot' in output:
            losses['rot'] += checkpoint(self.crit_rot, output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
                                        batch['rotres'])
            # losses['rot'] += self.crit_rot(
            #     output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
            #     batch['rotres'])

        losses['tot'] = 0
        for head in heads:
            losses['tot'] += losses[head]

        return losses['tot'], losses
