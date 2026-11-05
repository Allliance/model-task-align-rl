"""CPU checks for rollout grouping, soft weights and exact small-set recovery."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from .weights import guidance_weights,prepare_guided_batch


class TestGuidance(unittest.TestCase):
    def inputs(self):
        # First large group conflicts; second agrees; third is inactive;
        # fourth conflicts but is a trusted small-pool group.
        a=torch.tensor([-1.,1.]*4)
        adv=torch.cat([a,a,torch.zeros(8),a])[:,None].repeat(1,3)
        delta=torch.cat([-a,a,a,-a])[:,None].repeat(1,3)
        mask=torch.ones_like(adv);mask[:,-1]=0
        # Padding must never affect the trust signal.
        delta[:,-1]=1e8
        ref=delta;old=torch.zeros_like(delta)
        uid=np.array([f'g{i//8}' for i in range(32)],dtype=object)
        pools=['large']*24+['small']*8
        return adv,ref,old,mask,uid,pools

    def test_adaptive_has_expected_sign_and_never_suppresses_inactive_or_d1(self):
        pg,kl,m=guidance_weights(*self.inputs(),'guide-adaptive-kl')
        torch.testing.assert_close(kl[:8],torch.full((8,),10.))
        torch.testing.assert_close(kl[8:],torch.ones(24))
        torch.testing.assert_close(pg,torch.ones(32))
        self.assertEqual(m['guided/active_prompt_fraction'],.75)

    def test_soft_pg_preserves_baseline_kl(self):
        pg,kl,m=guidance_weights(*self.inputs(),'guide-soft-pg')
        torch.testing.assert_close(pg[:8],torch.full((8,),.1))
        torch.testing.assert_close(pg[8:],torch.ones(24))
        torch.testing.assert_close(pg*kl,torch.ones(32))

    def test_group_order_is_independent_of_balancing(self):
        values=self.inputs();pg,kl,_=guidance_weights(*values,'guide-adaptive-kl')
        order=torch.randperm(32,generator=torch.Generator().manual_seed(314))
        shuffled=tuple(v[order] for v in values[:4])+(values[4][order.numpy()], [values[5][i] for i in order.tolist()])
        p2,k2,_=guidance_weights(*shuffled,'guide-adaptive-kl')
        torch.testing.assert_close(p2,pg[order]);torch.testing.assert_close(k2,kl[order])

    def test_teacher_current_not_teacher_base_and_complete_groups(self):
        values=list(self.inputs());values[2]=values[1].clone()
        _,kl,_=guidance_weights(*values,'guide-adaptive-kl')
        torch.testing.assert_close(kl,torch.ones(32))
        values=list(self.inputs());values[4][0]='broken'
        with self.assertRaisesRegex(ValueError,'complete rollout group'):
            guidance_weights(*values,'guide-adaptive-kl')

    def test_only_pool_is_read_from_example_metadata(self):
        class PoolOnly(dict):
            def get(self,key,*args):
                if key!='pool':raise AssertionError('Oracle metadata accessed: '+key)
                return super().get(key,*args)
        adv,ref,old,mask,uids,pools=self.inputs()
        data=SimpleNamespace(batch={'advantages':adv,'ref_log_prob':ref,'old_log_probs':old,'response_mask':mask},
            non_tensor_batch={'uid':uids,'extra_info':[PoolOnly(pool=pool) for pool in pools]})
        config=SimpleNamespace(guided=SimpleNamespace(condition='guide-soft-pg'),
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=8)))
        prepare_guided_batch(data,config)
        torch.testing.assert_close(data.batch['sample_weights']*data.batch['kl_weights'],torch.ones(32))


class TestSampling(unittest.TestCase):
    def test_d1_can_oversample_and_resume_across_epoch_boundary(self):
        from .sampling import PlanDataLoader,PlanSampler
        loader=lambda:PlanDataLoader(dataset=list(range(156)),batch_size=160,num_workers=0,drop_last=True,
            collate_fn=lambda x:x,sampler=PlanSampler(list(range(156)),batches=3))
        a=loader();it=iter(a);first=next(it)
        self.assertEqual(len(first),160);self.assertTrue(all(0<=i<156 for i in first))
        state=copy.deepcopy(a.state_dict());expected=[next(it),next(it)]
        with self.assertRaises(StopIteration):next(it)
        expected.append(next(iter(a)))
        b=loader();b.load_state_dict(state);it=iter(b);actual=[next(it),next(it)]
        with self.assertRaises(StopIteration):next(it)
        actual.append(next(iter(b)))
        self.assertEqual(actual,expected)
        self.assertTrue(torch.equal(a.sampler.generator.get_state(),b.sampler.generator.get_state()))
        wrong=PlanDataLoader(dataset=list(range(157)),batch_size=160,num_workers=0,drop_last=True,
            sampler=PlanSampler(list(range(157)),batches=3))
        with self.assertRaisesRegex(ValueError,'protocol mismatch'):
            wrong.load_state_dict(state)


if __name__=='__main__':
    unittest.main(verbosity=2)
