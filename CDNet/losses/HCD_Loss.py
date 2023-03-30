import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class HCD_loss(nn.Module):
    def __init__(self):
        super(HCD_loss, self).__init__()

        self.batch_size = 64
        self.balanced_parameter = 0.005

        self.bn = nn.BatchNorm1d(128, affine=False)

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, features, labels=None, mask=None, z1=None, z2=None):
        device = torch.device('cuda:0')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('not match')
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('not match')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), 0.07)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        pull_push_loss = - mean_log_prob_pos
        pull_push_loss = pull_push_loss.view(anchor_count, batch_size).mean()

        c = self.bn(z1).T @ self.bn(z2)
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        decouple_loss = on_diag + self.balanced_parameter * off_diag

        hcdloss = pull_push_loss + decouple_loss
        return hcdloss


if __name__ == "__main__":

    pass