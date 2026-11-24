import os
import unittest
from unittest.mock import Mock, patch

from verl.utils.tracking import Tracking


class ResumeLoggingTest(unittest.TestCase):
    def test_replayed_iterations_are_preserved_with_attempt_identity(self):
        tracking = Tracking.__new__(Tracking)
        tracking._synlogic_run = True
        logger = Mock()
        tracking.logger = {"wandb": logger}
        with patch.dict(os.environ, {"SYNLOGIC_ATTEMPT": "2"}):
            tracking.log({"accuracy": .5}, step=12)
            tracking.log({"accuracy": .6}, step=10)
        calls = logger.log.call_args_list
        self.assertEqual([call.kwargs["data"]["trainer/iteration"] for call in calls], [12, 10])
        self.assertTrue(all(call.kwargs["data"]["trainer/attempt"] == 2 for call in calls))
        self.assertTrue(all("step" not in call.kwargs for call in calls))
        tracking.logger = {}
