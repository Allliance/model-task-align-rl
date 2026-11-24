import torch
from verl.trainer.ppo.core_algos import agg_loss,compute_policy_loss,kl_penalty
old=torch.full((3,4),-1.0,dtype=torch.float64)
logp=torch.tensor([[-1.1,-1.05,-1.02,-1.0],[-.9,-.95,-1.0,-1.04],[-1.01,-1.02,-1.03,-1.04]],dtype=torch.float64,requires_grad=True)
adv=torch.tensor([[1.]*4,[-1.]*4,[.7]*4],dtype=torch.float64)
mask=torch.tensor([[1.,1.,0.,0.],[1.,1.,1.,1.],[1.,1.,1.,0.]],dtype=torch.float64)
w=torch.tensor([50.03205128205,.505047236,.505047236],dtype=torch.float64)
base=compute_policy_loss(old,logp,adv,mask,cliprange=.2)[0]
ones=compute_policy_loss(old,logp,adv,mask,cliprange=.2,sample_weights=torch.ones(3,dtype=torch.float64))[0]
assert torch.equal(base,ones)
loss=compute_policy_loss(old,logp,adv,mask,cliprange=.2,sample_weights=w)[0]
manual=(-(logp-old).exp()*adv*mask*w[:,None]).sum()/(mask.sum()+1e-8)
assert torch.allclose(loss,manual,atol=1e-12,rtol=1e-12)
a=torch.autograd.grad(loss,logp,retain_graph=True)[0];b=torch.autograd.grad(manual,logp)[0]
assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
x=torch.randn(1,4,requires_grad=True,dtype=torch.float64)
assert torch.allclose(agg_loss(x,torch.ones_like(x),'token-mean',torch.tensor([50.])),50*agg_loss(x,torch.ones_like(x),'token-mean'))
print('Weighted PPO gradients match the explicit objective; unit weights preserve the original result; single-pool microbatches retain their scaling.')
