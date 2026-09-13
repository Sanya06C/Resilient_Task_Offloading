import pickle
import unittest

import numpy as np

from MEC_Env import MEC
from accounting import EpisodeAccounting
from stress_aware_selector import StressAwareSelector


def make_env(size=4.0, density=.297, capacity=14.0, max_delay=10):
    env = MEC(2, 2, 20, 1, max_delay)
    env.accounting = EpisodeAccounting("power_time_v1")
    env.tran_cap_ue[:] = capacity * env.duration
    sizes = np.zeros((20, 2))
    densities = np.zeros_like(sizes)
    sizes[0, 0] = size
    densities[0, 0] = density
    env.reset(sizes, densities)
    return env


class StressAwareSelectorTests(unittest.TestCase):
    def test_selects_feasible_lower_energy_edge_and_retains_equal_original_edge(self):
        env = make_env()
        selector = StressAwareSelector()
        selected, decision = selector.select(env, 0, 0)
        self.assertEqual(selected, 1)
        self.assertEqual(decision["reason"], "override_lower_ue_energy")
        self.assertTrue(all(x["feasible"] for x in decision["estimates"]))
        self.assertLess(decision["estimates"][1]["ue_energy"],
                        decision["estimates"][0]["ue_energy"])

        selected, decision = selector.select(env, 0, 2)
        self.assertEqual(selected, 2)
        self.assertEqual(decision["reason"], "retain_original_minimum")

    def test_deadline_feasibility_blocks_slow_transmission(self):
        env = make_env(size=6.9, density=.397, capacity=6.326878361484239)
        selected, decision = StressAwareSelector().select(env, 0, 0)
        self.assertEqual(selected, 0)
        self.assertFalse(decision["estimates"][0]["feasible"])
        self.assertFalse(decision["estimates"][1]["feasible"])
        self.assertFalse(decision["estimates"][2]["feasible"])
        self.assertEqual(decision["reason"], "retain_no_feasible_action")

    def test_completed_transmission_enters_edge_with_full_workload(self):
        env = make_env(size=4.0, density=.297)
        selector = StressAwareSelector()
        tx = selector._simulate_transmission(env, 0, 0)
        candidate_event = next(x for x in tx["completions"]
                               if x["job"]["candidate"])
        self.assertAlmostEqual(candidate_event["job"]["remaining"], 4.0)
        edge = selector._simulate_edge(
            env, 0, 0, tx["completions"], tx["completion_slot"])
        self.assertTrue(edge["feasible"])
        self.assertGreaterEqual(edge["completion_slot"], tx["completion_slot"] + 1)

    def test_current_queue_can_change_best_action(self):
        env = make_env(size=4.0, density=.297)
        # A known local backlog makes local completion infeasible while the
        # uncongested edges remain feasible.
        old = {"TIME": 0, "SIZE": 7.0, "DENS": .397, "REMAIN": 7.0,
               "EDGE": -1, "DIV": 0, "UE_ID": 0, "TASK_ID": 99}
        env.local_process_task[0] = old
        selected, decision = StressAwareSelector().select(env, 0, 0)
        self.assertIn(selected, (1, 2))
        self.assertFalse(decision["estimates"][0]["feasible"])
        self.assertTrue(decision["estimates"][selected]["feasible"])

    def test_estimation_does_not_mutate_environment_state(self):
        env = make_env()
        before = pickle.dumps({
            "local": env.local_process_task,
            "tx": env.local_transmit_task,
            "edge": env.edge_process_task,
            "local_q": [list(q.queue) for q in env.ue_computation_queue],
            "tx_q": [list(q.queue) for q in env.ue_transmission_queue],
            "edge_q": [[list(q.queue) for q in row]
                       for row in env.edge_computation_queue],
        })
        StressAwareSelector().select(env, 0, 1)
        after = pickle.dumps({
            "local": env.local_process_task,
            "tx": env.local_transmit_task,
            "edge": env.edge_process_task,
            "local_q": [list(q.queue) for q in env.ue_computation_queue],
            "tx_q": [list(q.queue) for q in env.ue_transmission_queue],
            "edge_q": [[list(q.queue) for q in row]
                       for row in env.edge_computation_queue],
        })
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
