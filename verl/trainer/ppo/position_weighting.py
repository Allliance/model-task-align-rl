"""PW-OPSD Eq.5 Moderate schedule; baseline OPD global-token adaptation.

Only valid response tokens determine rank/length, including a valid EOS.
Compute on complete response rows before actor minibatch/microbatch splitting.
No clean labels or teacher distributions enter this fixed detached schedule.
"""
import torch


@torch.no_grad()
def response_position_weights(response_mask):
    if response_mask.ndim != 2 or not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("Expected a binary [rows, full response width] mask")
    mask = response_mask.bool()
    valid = mask.to(torch.float32)
    lengths = valid.sum(-1, keepdim=True)
    relative = (valid.cumsum(-1) - 0.5) / lengths.clamp_min(1)
    weights = 0.25 + 0.75 * torch.sigmoid((relative - 0.30) / 0.10)
    return torch.where(mask, weights, 0.0)
