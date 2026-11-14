"""EOPD Eq.9/10 (arXiv:2603.07079): additive, entropy-gated top16 FKL.

The actor supplies CURRENT full-vocabulary-normalized student log probabilities.
Teacher top16 probabilities are renormalized here; teacher entropy is full-vocab.
RKL clipping and global original-token normalization remain in the native actor.
The paper's strict >.8 gate is intentional (published code uses >=.8).
"""
import torch


def forward_kl(student_lp, teacher_lp, teacher_entropy, response_mask):
    assert student_lp.shape == teacher_lp.shape and student_lp.shape[-1] == 16
    assert student_lp.shape[:-1] == teacher_entropy.shape == response_mask.shape
    mask=response_mask.bool();n=mask.sum()
    assert n>0, 'EOPD requires original valid response tokens'
    student=student_lp[mask].float()
    teacher=teacher_lp[mask].detach().float()
    entropy=teacher_entropy[mask].detach().float()
    assert torch.isfinite(student).all() and torch.isfinite(teacher).all() and torch.isfinite(entropy).all()
    assert (student<=1e-5).all() and (teacher<=1e-5).all() and (entropy>=-1e-5).all()
    mass=teacher.exp().sum(-1)
    assert (mass>0).all() and (mass<=1.0001).all(), 'Teacher statistics must be full-normalized'
    target_lp=teacher-torch.logsumexp(teacher,-1,keepdim=True)
    per_token=(target_lp.exp()*(target_lp-student)).sum(-1)
    gate=entropy>.8
    # No tail bucket, per-vocabulary clamp, student top-k renormalization,
    # FKL PPO clipping or selected-only denominator.
    loss=(per_token*gate).sum()/n
    assert torch.isfinite(loss)
    return loss,{'entropy_sum':entropy.sum(),'retained_mass_sum':mass.sum(),
                 'gated_tokens':gate.sum()}
