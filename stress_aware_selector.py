"""Decision-time selective offloading for a frozen QECO policy.

The selector reads only the arriving task, current queues, deadline, current
capacities, and configured powers. It does not read future workloads, terminal
outcomes, accounting arrays, or learned model internals.
"""
from copy import deepcopy
import math

import numpy as np


class StressAwareSelector:
    ACTION_NAMES = {0: "local", 1: "edge_0", 2: "edge_1"}

    def __init__(self, energy_tolerance=1e-12):
        self.energy_tolerance = energy_tolerance
        self.reset_episode()

    def reset_episode(self):
        self.decisions = []

    @staticmethod
    def _valid_task(task):
        return task is not None and float(task.get("SIZE", 0)) > 0

    @staticmethod
    def _active_task(task):
        return task is not None and np.isfinite(task.get("REMAIN", np.nan)) \
            and float(task["REMAIN"]) > 0

    @staticmethod
    def _copy_job(task, candidate=False):
        edge_value = task.get("EDGE", -1)
        edge = int(edge_value) if np.isfinite(edge_value) else -1
        return {
            "arrival": int(task["TIME"]),
            "density": float(task["DENS"]),
            "remaining": float(task.get("REMAIN", task["SIZE"])),
            "size": float(task["SIZE"]),
            "edge": edge,
            "candidate": bool(candidate),
        }

    @staticmethod
    def _queued(queue_object):
        # QECO is single-threaded; this creates a read-only decision-time copy.
        with queue_object.mutex:
            return [deepcopy(task) for task in list(queue_object.queue)]

    def _existing_jobs(self, active, queue_object):
        jobs = []
        if self._active_task(active):
            jobs.append(self._copy_job(active))
        jobs.extend(self._copy_job(task) for task in self._queued(queue_object)
                    if self._valid_task(task))
        return jobs

    @staticmethod
    def _candidate(env, ue, edge=-1):
        t = env.time_count
        return {
            "TIME": t,
            "SIZE": float(env.arrive_task_size[t, ue]),
            "DENS": float(env.arrive_task_dens[t, ue]),
            "EDGE": edge,
        }

    @staticmethod
    def _serve_fifo(jobs, start_slot, last_slot, capacity_for_job,
                    collect_completions=False):
        """Simulate known FIFO work. Service occurs once per integer slot."""
        jobs = deepcopy(jobs)
        slot = int(start_slot)
        completions = []
        candidate_result = None
        while slot <= last_slot and jobs:
            job = jobs[0]
            # All QECO tasks use the same max-delay horizon. The caller supplies
            # last_slot for the new task, and an absolute deadline per old job.
            deadline = job["arrival"] + (last_slot - start_slot)
            if slot > deadline:
                jobs.pop(0)
                if job["candidate"]:
                    candidate_result = {"feasible": False}
                    break
                continue
            capacity = float(capacity_for_job(job))
            if not np.isfinite(capacity) or capacity <= 0:
                raise ValueError("Nonpositive/nonfinite predicted capacity")
            served = min(job["remaining"], capacity)
            fraction = served / capacity
            job["remaining"] -= served
            if job["remaining"] <= 1e-12:
                finished = jobs.pop(0)
                event = {
                    "completion_slot": slot,
                    "available_slot": slot + 1,
                    "edge": finished["edge"],
                    # A transmitted task enters edge computation with its full
                    # CPU workload. Transmission service must not carry its
                    # depleted REMAIN value into the edge forecast.
                    "job": {**finished, "remaining": finished["size"]},
                    "last_fraction": fraction,
                }
                if collect_completions:
                    completions.append(event)
                if finished["candidate"]:
                    candidate_result = {
                        "feasible": True,
                        "completion_slot": slot,
                        "last_fraction": fraction,
                    }
                    break
            elif slot == deadline:
                dropped = jobs.pop(0)
                if dropped["candidate"]:
                    candidate_result = {"feasible": False}
                    break
            slot += 1
        if candidate_result is None:
            candidate_result = {"feasible": False}
        candidate_result["completions"] = completions
        return candidate_result

    def estimate_local(self, env, ue):
        task = self._candidate(env, ue)
        candidate = self._copy_job(task, candidate=True)
        jobs = self._existing_jobs(
            env.local_process_task[ue], env.ue_computation_queue[ue])
        jobs.append(candidate)
        t = env.time_count
        deadline = t + env.max_delay - 1
        result = self._serve_fifo(
            jobs, t, deadline,
            lambda job: env.comp_cap_ue[ue] / job["density"])
        active_seconds = (task["SIZE"] /
                          (env.comp_cap_ue[ue] / task["DENS"])) * env.duration
        result.update(
            action=0, action_name="local",
            ue_energy=float(env.ue_p_comp * active_seconds),
            predicted_delay_slots=(
                result.get("completion_slot", deadline + 1) - t + 1))
        if not result["feasible"]:
            result["objective_energy"] = float("inf")
        else:
            result["objective_energy"] = result["ue_energy"]
        result.pop("completions", None)
        return result

    def _simulate_transmission(self, env, ue, candidate_edge):
        task = self._candidate(env, ue, edge=candidate_edge)
        candidate = self._copy_job(task, candidate=True)
        jobs = self._existing_jobs(
            env.local_transmit_task[ue], env.ue_transmission_queue[ue])
        jobs.append(candidate)
        t = env.time_count
        deadline = t + env.max_delay - 1
        result = self._serve_fifo(
            jobs, t, deadline,
            lambda job: env.tran_cap_ue[ue, job["edge"]],
            collect_completions=True)
        result["candidate"] = candidate
        return result

    def _simulate_edge(self, env, ue, edge, transmission_events,
                       candidate_tx_completion):
        t = env.time_count
        last_slot = t + env.max_delay - 1
        existing = self._existing_jobs(
            env.edge_process_task[ue][edge],
            env.edge_computation_queue[ue][edge])
        events = sorted(
            [event for event in transmission_events if event["edge"] == edge],
            key=lambda event: event["available_slot"])
        ue_already_active = bool(existing)
        active_users = max(float(env.edge_ue_m[edge]) +
                           (0.0 if ue_already_active else 1.0), 1.0)
        jobs = existing
        slot = t
        candidate_started = None
        candidate_last_fraction = None
        while slot <= last_slot:
            while events and events[0]["available_slot"] <= slot:
                jobs.append(events.pop(0)["job"])
            while jobs and slot > jobs[0]["arrival"] + env.max_delay - 1:
                expired = jobs.pop(0)
                if expired["candidate"]:
                    return {"feasible": False, "active_users": active_users}
            if jobs:
                job = jobs[0]
                if job["candidate"] and candidate_started is None:
                    candidate_started = slot
                capacity = (env.comp_cap_edge[edge] /
                            job["density"] / active_users)
                served = min(job["remaining"], capacity)
                fraction = served / capacity
                job["remaining"] -= served
                deadline = job["arrival"] + env.max_delay - 1
                if job["remaining"] <= 1e-12:
                    finished = jobs.pop(0)
                    if finished["candidate"]:
                        candidate_last_fraction = fraction
                        idle_slot_equivalent = (
                            slot - candidate_tx_completion - 1 + fraction)
                        return {
                            "feasible": True,
                            "completion_slot": slot,
                            "last_fraction": candidate_last_fraction,
                            "idle_slot_equivalent": max(idle_slot_equivalent, 0.0),
                            "active_users": active_users,
                            "edge_start_slot": candidate_started,
                        }
                elif slot == deadline:
                    dropped = jobs.pop(0)
                    if dropped["candidate"]:
                        return {"feasible": False, "active_users": active_users}
            slot += 1
        return {"feasible": False, "active_users": active_users}

    def estimate_edge(self, env, ue, edge):
        tx = self._simulate_transmission(env, ue, edge)
        t = env.time_count
        action = edge + 1
        task = self._candidate(env, ue, edge=edge)
        tx_seconds = (task["SIZE"] / env.tran_cap_ue[ue, edge]) * env.duration
        tx_energy = float(env.ue_p_tran * tx_seconds)
        result = {
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "tx_energy": tx_energy,
            "idle_energy": float("inf"),
            "ue_energy": float("inf"),
            "objective_energy": float("inf"),
            "feasible": False,
            "tx_feasible": bool(tx["feasible"]),
        }
        if not tx["feasible"]:
            result["predicted_delay_slots"] = env.max_delay + 1
            return result
        edge_result = self._simulate_edge(
            env, ue, edge, tx["completions"], tx["completion_slot"])
        result["edge_active_users"] = edge_result["active_users"]
        if not edge_result["feasible"]:
            result["predicted_delay_slots"] = env.max_delay + 1
            return result
        idle_energy = float(
            env.ue_p_idle * env.duration * edge_result["idle_slot_equivalent"])
        result.update(
            feasible=True,
            idle_energy=idle_energy,
            ue_energy=tx_energy + idle_energy,
            objective_energy=tx_energy + idle_energy,
            tx_completion_slot=int(tx["completion_slot"]),
            edge_completion_slot=int(edge_result["completion_slot"]),
            predicted_delay_slots=int(edge_result["completion_slot"] - t + 1),
        )
        return result

    def select(self, env, ue, original_action):
        original_action = int(original_action)
        size = float(env.arrive_task_size[env.time_count, ue])
        density = float(env.arrive_task_dens[env.time_count, ue])
        if size <= 0:
            return original_action, None
        if density <= 0:
            raise ValueError("Arriving task must have positive computation density")
        estimates = [self.estimate_local(env, ue)]
        estimates.extend(self.estimate_edge(env, ue, edge)
                         for edge in range(env.n_edge))
        feasible = [item for item in estimates if item["feasible"]]
        chosen = original_action
        reason = "retain_no_feasible_action"
        if feasible:
            best_energy = min(item["objective_energy"] for item in feasible)
            original = estimates[original_action]
            if original["feasible"] and original["objective_energy"] \
                    <= best_energy + self.energy_tolerance:
                reason = "retain_original_minimum"
            else:
                best = min(
                    (item for item in feasible
                     if item["objective_energy"] <= best_energy + self.energy_tolerance),
                    key=lambda item: item["action"])
                chosen = best["action"]
                reason = ("override_original_infeasible" if not original["feasible"]
                          else "override_lower_ue_energy")
        decision = {
            "time": int(env.time_count),
            "ue": int(ue),
            "size": size,
            "density": density,
            "effective_transmission_capacity": env.tran_cap_ue[ue].tolist(),
            "original_action": original_action,
            "selected_action": int(chosen),
            "overridden": bool(chosen != original_action),
            "reason": reason,
            "estimates": estimates,
        }
        self.decisions.append(decision)
        return int(chosen), decision
