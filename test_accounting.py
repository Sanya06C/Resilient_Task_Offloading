import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings
import numpy as np
from MEC_Env import MEC
from accounting import EpisodeAccounting

ROOT = Path(__file__).resolve().parent


def original_mec():
    spec = importlib.util.spec_from_file_location(
        "original_mec", ROOT / "accounting_baseline" / "original_sources" / "MEC_Env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MEC


class AccountingTests(unittest.TestCase):
    def run_episode(self, cls=MEC, max_delay=3):
        np.random.seed(19)
        env = cls(2, 2, 25, 1, max_delay)
        if cls is MEC:
            env.accounting = EpisodeAccounting()
        rng = np.random.RandomState(31)
        sizes = rng.uniform(1, 7, (25, 2)) * (rng.rand(25, 2) < .9)
        sizes[-5:] = 0
        densities = np.where(sizes > 0, .297, 0)
        obs, lstm = env.reset(sizes.copy(), densities.copy())
        trajectory = [(obs.copy(), lstm.copy())]
        actions = rng.randint(0, 3, size=(25, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for a in actions:
                obs, lstm, done = env.step(a)
                trajectory.append((obs.copy(), lstm.copy()))
        return env, trajectory

    def test_accounting_does_not_change_policy_inputs_outcomes_or_reward_inputs(self):
        changed, new_trace = self.run_episode()
        original, old_trace = self.run_episode(original_mec())
        for new, old in zip(new_trace, old_trace):
            for x, y in zip(new, old):
                np.testing.assert_equal(x, y)
        for name in ("process_delay", "process_delay_trans", "unfinish_task",
                     "ue_comp_energy", "ue_tran_energy", "edge_comp_energy", "ue_idle_energy",
                     "t_ue_comp", "t_ue_tran", "b_edge_comp"):
            np.testing.assert_equal(getattr(changed, name), getattr(original, name))
        row = changed.accounting.summarize(changed, 0)
        self.assertEqual(row["arrived_tasks"], row["local_tasks"] + row["offloaded_tasks"])
        self.assertEqual(row["arrived_tasks"], row["completed_tasks"] + row["deadline_violations"])
        self.assertAlmostEqual(row["total_system_energy"], row["ue_only_total_energy"] + row["edge_computation_energy"])
        self.assertGreater(row["policy_timing_mismatch_arrivals"], 0)

    def test_processed_and_transmitted_volume_conservation(self):
        env, _ = self.run_episode()
        np.testing.assert_allclose(env.ue_bit_processed, env.accounting.volumes["local"])
        np.testing.assert_allclose(env.ue_bit_transmitted, env.accounting.volumes["tx"])
        np.testing.assert_allclose(env.edge_bit_processed.sum(axis=2), env.accounting.volumes["edge"])

    def test_deadline_equality_is_success_and_queued_expiries_are_counted(self):
        env = MEC(2, 2, 6, 1, 2)
        env.accounting = EpisodeAccounting()
        sizes = np.zeros((6, 2)); sizes[0, 0] = .7
        sizes[:3, 1] = 7
        density = np.where(sizes > 0, .397, 0)
        env.reset(sizes, density)
        for _ in range(6):
            env.step(np.zeros(2))
        row = env.accounting.summarize(env, 0)
        self.assertEqual(env.process_delay[0, 0], 2)
        self.assertEqual(row["completed_tasks"], 1)
        self.assertEqual(row["deadline_violations"], 3)
        self.assertEqual(row["average_latency_slots"], 2)

    def test_empty_episode(self):
        env = MEC(2, 2, 4, 1, 2)
        env.accounting = EpisodeAccounting()
        env.reset(np.zeros((4, 2)), np.zeros((4, 2)))
        for _ in range(4):
            env.step(np.zeros(2))
        row = env.accounting.summarize(env, 0)
        self.assertEqual(row["total_system_energy"], 0)
        self.assertIsNone(row["average_latency_seconds"])
        self.assertIsNone(row["offloaded_fraction"])

    def test_energy_logging_sums_all_edges(self):
        tree = ast.parse((ROOT / "D3QN.py").read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "do_store_energy")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "energy_logger", "exec"), namespace)
        obj = SimpleNamespace(n_time=1, energy_store=[])
        namespace["do_store_energy"](obj, 0, 0, 2, 4, np.array([1, 3]), np.array([.5, .7]))
        self.assertAlmostEqual(obj.energy_store[0][0], 11.2)

    def test_proposed_power_time_model_bounds_and_reset(self):
        env, _ = self.run_episode()
        env.accounting.energy_model = "power_time_v1"
        row = env.accounting.summarize(env, 0)
        self.assertGreaterEqual(row["ue_idle_energy"], 0)
        self.assertLessEqual(row["ue_idle_energy"], env.n_time * env.n_ue * env.duration * env.ue_p_idle)
        self.assertLessEqual(row["edge_computation_energy"], env.n_time * env.n_edge * env.duration * env.edge_p_comp)
        env.reset(np.zeros((25, 2)), np.zeros((25, 2)))
        self.assertEqual(env.accounting.local_fraction.sum(), 0)
        self.assertTrue(np.all(env.accounting.actions == -1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
