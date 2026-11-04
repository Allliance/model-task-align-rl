"""Regression checks on the native checkpoint predicates without loading GPUs."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class CheckpointAfterResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'src/dapo_ray_trainer.py'
        tree = ast.parse(source.read_text())
        fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        assignments = [n for n in ast.walk(fit) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                       and any(isinstance(t, ast.Name) and t.id == 'first_step_after_load'
                               for t in getattr(n, 'targets', [getattr(n, 'target', None)]))]
        if len(assignments) != 1 or assignments[0] not in fit.body:
            raise AssertionError('First recovery step must be assigned once outside the loop')
        initial = assignments[0]
        restored = next(n for n in fit.body if isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                        and n.value.func.attr == '_load_checkpoint')
        increment = next(n for n in fit.body if isinstance(n, ast.AugAssign)
                         and isinstance(n.target, ast.Attribute) and n.target.attr == 'global_steps')
        if not fit.body.index(restored) < fit.body.index(initial) < fit.body.index(increment):
            raise AssertionError('First recovery step must follow restore and precede counter increment')
        first = next(n for n in ast.walk(fit) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'first_recovery_save' for t in n.targets))
        save = next(n for n in ast.walk(fit) if isinstance(n, ast.If)
                    and 'first_recovery_save' in ast.unparse(n.test)
                    and 'save_freq' in ast.unparse(n.test))
        cls.initial = compile(ast.fix_missing_locations(ast.Expression(initial.value)), str(source), 'eval')
        cls.first = compile(ast.fix_missing_locations(ast.Expression(first.value)), str(source), 'eval')
        cls.save = compile(ast.fix_missing_locations(ast.Expression(save.test)), str(source), 'eval')

    def should_save(self, restored, step, *, enabled=True, draining=False, last=False):
        actor = SimpleNamespace(global_steps=restored,
                                config=SimpleNamespace(trainer=SimpleNamespace(save_freq=200)))
        scope = {'self': actor, 'os': SimpleNamespace(environ={'SYNLOGIC_RUN_ROOT': '/run'} if enabled else {})}
        scope['first_step_after_load'] = eval(self.initial, scope)
        actor.global_steps = step
        scope.update(draining=draining, is_last_step=last)
        scope['first_recovery_save'] = eval(self.first, scope)
        return eval(self.save, scope)

    def test_fresh_and_resumed_first_updates(self):
        self.assertTrue(self.should_save(0, 1))
        self.assertTrue(self.should_save(40, 41))
        self.assertTrue(self.should_save(200, 201))
        self.assertFalse(self.should_save(40, 42))

    def test_periodic_final_and_allocation_drain_saves(self):
        self.assertTrue(self.should_save(40, 200))
        self.assertFalse(self.should_save(40, 199))
        self.assertTrue(self.should_save(40, 970, last=True))
        self.assertTrue(self.should_save(40, 234, draining=True))

    def test_legacy_runs_keep_existing_first_save_behavior(self):
        self.assertFalse(self.should_save(0, 1, enabled=False))
        self.assertFalse(self.should_save(40, 41, enabled=False))
        self.assertTrue(self.should_save(40, 200, enabled=False))


if __name__ == '__main__':
    unittest.main()
