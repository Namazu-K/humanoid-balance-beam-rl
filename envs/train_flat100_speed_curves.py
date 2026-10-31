"""Signed logarithmic goal-closing speed trial; no linear progress or fixed-speed target (not a stability-certified CLF-RL reproduction). Place in /root/rope beside the original dependencies.
Requires --out; use --source for a new experiment or --resume for continuation.
Upright locomotion experiment: reward and fall termination are changed; physics and 366-D observation retained.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.modules["triton"] = None  # local CPU/PPO.load workaround

import argparse
import csv
import contextlib
import io
import json
import time
import signal
from collections import deque
from multiprocessing import freeze_support
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parent
GROUND_XML = ROOT / "results/flat100_groundfix/flat_ground_solid.xml"
GROUND_MODEL = ROOT / "results/flat100_groundfix/model_latest.zip"
REWARD_VERSION = "speed_curves_v8"
STOP_REQUESTED = False
STANDING_MODEL = ROOT / "results/flat100_speed_straight/n16_from34502656_5m/model_latest.zip"


def request_stop(signum=None, frame=None):
    # Parent only: do not raise in the middle of writing a checkpoint.
    global STOP_REQUESTED
    STOP_REQUESTED = True


def should_stop(path):
    return STOP_REQUESTED or (path is not None and Path(path).exists())


def posture_deadband(height_error, roll, pitch, height_band, angle_band):
    value = np.array([height_error, roll, pitch], dtype=float)
    band = np.array([height_band, angle_band, angle_band], dtype=float)
    scale = np.array([0.15, 0.35, 0.35])
    return np.sign(value) * np.maximum(np.abs(value) - band, 0.0) / scale



class LogVelocityReward:
    """Log speed with a posture-only candidate CLF; no stability certificate.

    P solves CARE for a normalized double integrator with Q=I and R=1.
    The actual underactuated humanoid is NOT assumed to be that integrator.
    """
    P = np.array([[np.sqrt(3.0), 1.0], [1.0, np.sqrt(3.0)]])

    def __init__(self, decay_weight, speed_weight=2.0, log_scale=0.25, speed_curve="log"):
        self.log_scale = log_scale
        self.speed_curve = speed_curve
        self.decay_weight = decay_weight
        self.speed_weight = speed_weight
        self.previous_value = None

    def value(self, error, error_rate, vx, vy, lateral_error=0.0):
        state = np.stack([error, 0.25 * error_rate], axis=-1)
        posture = float(np.mean(np.einsum("...i,ij,...j->...", state, self.P, state)))
        # No fixed target-speed or sideways-velocity tracking remains.
        return posture + lateral_error ** 2, posture, 0.0

    def reward(self, error, error_rate, vx, vy, upright, dt, action, previous_action, slip, lateral_error=0.0, closing_speed=0.0):
        value, posture, velocity = self.value(error, error_rate, vx, vy, lateral_error)
        violation = 0.0 if self.previous_value is None else max(
            0.0, (value - self.previous_value) / dt + self.previous_value)
        self.previous_value = value
        # Signed log of actual goal-distance reduction / dt.
        # Backward motion is penalized without posture attenuation.
        if self.speed_curve == "log":
            velocity_reward = self.speed_weight * np.sign(closing_speed) * np.log1p(abs(closing_speed) / self.log_scale)
        else:
            # Normalized logistic: midpoint 5 m/s, width 2 m/s, cap 10 m/s.
            # A linear backward penalty avoids a backward saturation loophole.
            low = 1.0 / (1.0 + np.exp(2.5))
            high = 1.0 - low
            if closing_speed < 0:
                velocity_reward = self.speed_weight * low * (1.0-low) / (2.0*(high-low)) * closing_speed
            else:
                v = min(float(closing_speed), 10.0)
                velocity_reward = self.speed_weight * (1.0/(1.0+np.exp(-(v-5.0)/2.0))-low)/(high-low)
        quality = float(np.exp(-0.5 * np.mean(np.square(error))))
        tracking = float(velocity_reward) * (quality if velocity_reward > 0 else 1.0)
        terms = {
            "track": dt * tracking,
            "decay": -dt * self.decay_weight * np.clip(violation / 10.0, 0.0, 1.0),
            "slip": -dt * 0.5 * min(slip / 0.25, 4.0),
            "effort": -dt * 0.02 * float(np.mean(np.square(action))),
            "smooth": -dt * 0.02 * float(np.mean(np.square(action - previous_action))),
        }
        return float(sum(terms.values())), value, violation, terms


class AlternatingSteps:
    """Diagnostics only: no bonus for time-based or contact-only foot tapping."""
    def __init__(self, foot_x, com_x):
        self.air = [0.0, 0.0]
        self.last_contact = [True, True]
        self.last_side = None
        self.records = list(foot_x)
        self.com_record = com_x
        self.count = 0

    def update(self, contacts, foot_x, com_x, upright, dt):
        for side in range(2):
            old_air = self.air[side]
            touchdown = contacts[side] and not self.last_contact[side]
            if (touchdown and old_air >= 0.08 and upright
                    and side != self.last_side
                    and foot_x[side] >= self.records[side] + 0.08
                    and com_x >= self.com_record + 0.03):
                self.count += 1
                self.last_side = side
                self.records[side] = foot_x[side]
                self.com_record = com_x
            self.air[side] = 0.0 if contacts[side] else old_air + dt
            self.last_contact[side] = contacts[side]
        return self.count


class StallTracker:
    """Optional watchdog. Disabled when window=0; no recovery-history state."""
    def __init__(self, window=0.0, distance=0.10, startup=4.0):
        self.window, self.distance, self.startup = window, distance, startup
        self.reset()

    def reset(self, initial_record=0.0):
        self.reference = float(initial_record)
        self.last_progress_time = 0.0

    def update(self, elapsed, best_com, low_posture=False):
        if best_com >= self.reference + self.distance - 1e-9:
            self.reference = best_com
            self.last_progress_time = elapsed
        clock = max(0.0, elapsed - max(self.startup, self.last_progress_time))
        return self.window > 0 and clock >= self.window, clock


def configure_existing_env(out, args):
    """Each spawned worker independently installs the same environment class."""
    import train_flat100_groundfix_camera as cfg

    runner = cfg.runner
    runner.OUT = Path(out)
    if not GROUND_XML.is_file():
        raise FileNotFoundError(
            f"平地XMLが見つかりません: {GROUND_XML}\n"
            "先に train_flat100_groundfix_camera.py --diagnose を実行してください"
        )
    # Do not regenerate physical XML from several processes at once.
    runner.beam_xml = lambda _width: GROUND_XML

    if not getattr(runner.Beam110Env, "_progress_stall_v1", False):
        OriginalFlatEnv = runner.Beam110Env
        window, distance = args.stall_window, args.stall_distance
        startup = args.startup_grace

        class ProgressStallEnv(OriginalFlatEnv):
            _progress_stall_v1 = True

            def reset(self, *, seed=None, options=None):
                obs, info = super().reset(seed=seed, options=options)
                self.progress_tracker = StallTracker(
                    window=window, distance=distance,
                    startup=startup,
                )
                self.progress_tracker.reset(0.0)
                self.walk_highwater = 0.0
                self.walk_distance = 0.0
                self.fall_seconds = 0.0
                self.upright_steps = 0
                self.walk_steps = 0
                self.clf = LogVelocityReward(args.decay_weight, args.speed_weight, args.log_scale, args.speed_curve)
                self.prev_error = self.posture_error()
                self.prev_com_xy = np.array(self.base.data.subtree_com[self.torso_id, :2], copy=True)
                self.prev_action = np.zeros(self.action_space.shape, dtype=float)
                self.line_y = float(self.prev_com_xy[1])
                self.previous_goal_distance = float(args.goal_distance)
                self.path_length = 0.0
                self.lateral_abs_sum = 0.0
                self.max_lateral_abs = 0.0
                self.max_com_m = 0.0
                self.step_tracker = AlternatingSteps(self.foot_x(), self.com_x())
                self.metric_sums = dict(clf_value=0.0, clf_violation=0.0, slip=0.0,
                                       reward_track=0.0, reward_decay=0.0, reward_progress=0.0)
                return obs, info

            def posture_error(self):
                torso = self.base.data.body("torso")
                rotation = torso.xmat.reshape(3, 3)
                roll = np.arctan2(rotation[2, 1], rotation[2, 2])
                pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
                return posture_deadband(float(torso.xpos[2]) - self.base.initial_z,
                                        roll, pitch, args.posture_height_band,
                                        np.deg2rad(args.posture_angle_band))

            def foot_x(self):
                return [float(self.base.data.geom(side + "_foot").xpos[0])
                        for side in ("left", "right")]

            def foot_contacts_and_slip(self):
                import mujoco
                m, d = self.base.model, self.base.data
                foot_ids = [int(m.geom(side + "_foot").id) for side in ("left", "right")]
                contacts = [False, False]
                # Include all collision geoms rigidly attached to the foot body.
                foot_bodies = [int(m.geom_bodyid[g]) for g in foot_ids]
                for k in range(d.ncon):
                    c = d.contact[k]
                    a, b = int(c.geom1), int(c.geom2)
                    other = b if a == self.beam_id else a if b == self.beam_id else None
                    if other is not None:
                        for side, body in enumerate(foot_bodies):
                            if int(m.geom_bodyid[other]) == body:
                                contacts[side] = True
                speeds = []
                jac = np.zeros((3, m.nv))
                for k in range(d.ncon):
                    c = d.contact[k]
                    a, b = int(c.geom1), int(c.geom2)
                    other = b if a == self.beam_id else a if b == self.beam_id else None
                    if other is not None and int(m.geom_bodyid[other]) in foot_bodies:
                        # Contact-point velocity, not foot-centre velocity:
                        # normal rolling/pivoting is not necessarily slip.
                        mujoco.mj_jac(m, d, jac, None, c.pos, int(m.geom_bodyid[other]))
                        speeds.append(float(np.linalg.norm((jac @ d.qvel)[:2])))
                return contacts, float(np.mean(speeds)) if speeds else 0.0

            def walk_upright(self):
                torso = self.base.data.body("torso")
                height = float(torso.xpos[2])
                vertical = float(torso.xmat.reshape(3, 3)[2, 2])
                return (height >= float(self.base.initial_z) - args.height_drop
                        and vertical >= args.min_upright)

            def clearly_fallen(self):
                torso = self.base.data.body("torso")
                return (float(torso.xpos[2]) < self.base.initial_z - args.fall_height_drop
                        or float(torso.xmat.reshape(3, 3)[2, 2]) < args.fall_alignment)

            def step(self, action):
                action = np.clip(np.asarray(action, dtype=np.float32),
                                 self.action_space.low, self.action_space.high)
                obs, _, _, _, info = super().step(action)
                info = dict(info)
                dt = float(self.base.dt)
                upright = self.walk_upright()
                com = max(0.0, self.com_x() - self.initial_com)
                # Track ALL progress, so standing up after crawling earns no backpay.
                increment = max(0.0, com - self.walk_highwater)
                self.walk_highwater = max(self.walk_highwater, com)
                valid_progress = increment if upright else 0.0
                self.walk_distance += valid_progress
                self.fall_seconds = self.fall_seconds + dt if self.clearly_fallen() else 0.0
                self.walk_steps += 1
                self.upright_steps += int(upright)
                error = self.posture_error()
                error_rate = (error - self.prev_error) / dt
                com_xy = np.array(self.base.data.subtree_com[self.torso_id, :2], copy=True)
                vx, vy = (com_xy - self.prev_com_xy) / dt
                contacts, slip = self.foot_contacts_and_slip()
                lateral = float(com_xy[1] - self.line_y)
                self.path_length += float(np.linalg.norm(com_xy - self.prev_com_xy))
                self.lateral_abs_sum += abs(lateral)
                self.max_lateral_abs = max(self.max_lateral_abs, abs(lateral))
                self.max_com_m = max(self.max_com_m, float(com_xy[0] - self.initial_com))
                current_goal_distance = float(np.hypot(
                    com_xy[0] - self.initial_com - args.goal_distance, lateral))
                closing_speed = (self.previous_goal_distance - current_goal_distance) / dt
                shaping, clf_value, violation, terms = self.clf.reward(
                    error, error_rate, float(vx), float(vy), upright,
                    dt, np.asarray(action), self.prev_action, slip, lateral / args.line_scale, closing_speed)
                self.prev_error = error
                self.prev_com_xy = com_xy
                self.prev_action = np.array(action, copy=True)
                alternating_steps = self.step_tracker.update(
                    contacts, self.foot_x(), self.com_x(), upright, dt)
                # Continuous path penalty: no lateral boundary or off-course termination.
                line_cost = dt * args.line_weight * min((lateral / args.line_scale) ** 2, 4.0)
                self.previous_goal_distance = current_goal_distance
                # No distance-proportional reward or fixed-speed tracking is added.
                progress_reward = 0.0
                reward = shaping - args.time_cost * dt - line_cost
                for key, value in dict(clf_value=clf_value, clf_violation=violation,
                                       slip=slip, reward_track=terms["track"],
                                       reward_decay=terms["decay"], reward_progress=progress_reward).items():
                    self.metric_sums[key] += value
                terminated = False
                truncated = False
                reason = "continuing"
                success = False
                stalled, clock = self.progress_tracker.update(
                    float(info["elapsed_seconds"]), com, False)
                if not np.all(np.isfinite(obs)):
                    raise FloatingPointError("Non-finite observation from physics")
                if self.fall_seconds >= args.fall_grace:
                    terminated = True
                    reason = "fallen"
                    reward -= args.fall_penalty
                elif (np.hypot(self.com_x() - self.initial_com - args.goal_distance,
                               float(com_xy[1] - self.line_y)) <= args.goal_tolerance and upright):
                    terminated = True
                    reason = "goal"
                    success = True
                    reward += 10.0
                elif stalled:
                    terminated = True
                    reason = "stalled"
                    reward -= args.fall_penalty
                elif float(info["elapsed_seconds"]) >= self.seconds:
                    truncated = True
                    reason = "time_limit"
                info.update(
                    reason=reason, is_success=success,
                    net_forward_m=float(com_xy[0] - self.initial_com),
                    episode_speed_mps=float(com_xy[0] - self.initial_com) / max(float(info["elapsed_seconds"]), 1e-9),
                    mean_abs_lateral_m=self.lateral_abs_sum / self.walk_steps,
                    max_abs_lateral_m=self.max_lateral_abs,
                    path_length_m=self.path_length,
                    straightness=float(np.clip((com_xy[0] - self.initial_com) / max(self.path_length, 1e-9), 0.0, 1.0)),
                    alternating_steps=alternating_steps,
                    mean_clf_value=self.metric_sums["clf_value"] / self.walk_steps,
                    mean_clf_violation=self.metric_sums["clf_violation"] / self.walk_steps,
                    mean_foot_slip=self.metric_sums["slip"] / self.walk_steps,
                    reward_track_total=self.metric_sums["reward_track"],
                    reward_decay_total=self.metric_sums["reward_decay"],
                    reward_progress_total=self.metric_sums["reward_progress"],
                    walking_m=float(self.walk_distance),
                    upright_pct=100.0 * self.upright_steps / self.walk_steps,
                    fall_seconds=float(self.fall_seconds),
                    no_com_progress_seconds=float(clock),
                    com_progress_reference_m=float(self.progress_tracker.reference),
                    recovery_grace_used=False,
                )
                return obs, float(reward), terminated, truncated, info

        runner.Beam110Env = ProgressStallEnv

    # Correct camera is inherited from groundfix_camera.py.
    # Align the native TimeLimit with the requested outer duration.
    original_make = runner.make_env

    def make_aligned_env(width, seconds, stall_seconds):
        env = original_make(width, seconds, stall_seconds)
        wrapped = env.base.env
        while hasattr(wrapped, "env"):
            if isinstance(wrapped, __import__("gymnasium").wrappers.TimeLimit):
                wrapped._max_episode_steps = int(np.ceil(seconds / env.base.dt)) + 1
            wrapped = wrapped.env
        return env

    # Do not stack factory wrappers when configuring repeatedly in one process.
    if not getattr(runner.make_env, "_aligned_v2", False):
        make_aligned_env._aligned_v2 = True
        runner.make_env = make_aligned_env
    return runner


def make_worker(index, output_dir, seconds, seed, args):
    def _factory():
        torch.set_num_threads(1)
        runner = configure_existing_env(output_dir, args)
        env = runner.make_env(1.0, seconds, args.stall_window)
        env.action_space.seed(seed + index)
        return Monitor(
            env,
            filename=str(Path(output_dir) / f"worker_{index}.monitor.csv"),
            override_existing=not args.resume,
            info_keywords=(
                "reason", "distance_m", "com_max_x_m", "rear_foot_max_x_m",
                "low_posture_pct", "hand_contact_pct", "is_success",
                "no_com_progress_seconds", "recovery_grace_used", "walking_m", "upright_pct", "alternating_steps", "mean_clf_value", "mean_clf_violation", "mean_foot_slip", "net_forward_m", "mean_abs_lateral_m", "max_abs_lateral_m", "path_length_m", "straightness", "reward_progress_total", "episode_speed_mps",
            ),
        )
    return _factory


class EpisodeReporter(BaseCallback):
    """Aggregate completed training episodes between evaluations; no console spam."""
    keys = ("com_max_x_m", "net_forward_m", "walking_m", "episode_speed_mps", "mean_abs_lateral_m",
            "max_abs_lateral_m", "straightness", "alternating_steps")

    def __init__(self, stop_file=None):
        super().__init__()
        self.stop_file = stop_file
        self.reset_window()

    def reset_window(self):
        self.count = 0
        self.sums = {k: 0.0 for k in self.keys}
        self.successes = 0
        self.falls = 0

    def _on_step(self):
        if should_stop(self.stop_file):
            return False
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.count += 1
                for key in self.keys:
                    self.sums[key] += float(info[key])
                self.successes += int(info.get("is_success", False))
                self.falls += int(info.get("reason") == "fallen")
        return True

    def save_window(self, out, added, total):
        row = {"added_steps": added, "total_steps": total, "episodes": self.count,
               **{"mean_" + k: self.sums[k] / self.count if self.count else ""
                  for k in self.keys}, "goal_count": self.successes, "fallen_count": self.falls}
        path = out / "training_progress.csv"
        present = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not present:
                writer.writeheader()
            writer.writerow(row)
        self.reset_window()
        return row


def evaluate_and_record(runner, model, out, label, seconds, count, video, args):
    """Single-environment deterministic assessment; independent fixed seeds."""
    env = runner.make_env(1.0, seconds, args.stall_window)
    rows = []
    try:
        for i in range(count):
            if should_stop(args.stop_file):
                return None
            seed = args.eval_seed_start + i
            obs, _ = env.reset(seed=seed)
            while True:
                if should_stop(args.stop_file):
                    return None
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(action)
                if done or trunc:
                    break
            rows.append({
                "checkpoint": label,
                "steps": model.num_timesteps,
                "seed": seed,
                "front_m": float(info["distance_m"]),
                "walking_m": float(info["walking_m"]),
                "episode_speed_mps": float(info["episode_speed_mps"]),
                "net_forward_m": float(info["net_forward_m"]),
                "mean_abs_lateral_m": float(info["mean_abs_lateral_m"]),
                "max_abs_lateral_m": float(info["max_abs_lateral_m"]),
                "path_length_m": float(info["path_length_m"]),
                "straightness": float(info["straightness"]),
                "upright_pct": float(info["upright_pct"]),
                "alternating_steps": int(info["alternating_steps"]),
                "mean_clf_value": float(info["mean_clf_value"]),
                "mean_clf_violation": float(info["mean_clf_violation"]),
                "mean_foot_slip": float(info["mean_foot_slip"]),
                "reward_track_total": float(info["reward_track_total"]),
                "reward_decay_total": float(info["reward_decay_total"]),
                "reward_progress_total": float(info["reward_progress_total"]),
                "com_m": float(info["com_max_x_m"]),
                "rear_foot_m": float(info["rear_foot_max_x_m"]),
                "seconds": float(info["elapsed_seconds"]),
                "reason": str(info["reason"]),
                "low_posture_pct": float(info["low_posture_pct"]),
                "hand_contact_pct": float(info["hand_contact_pct"]),
                "last_no_progress_s": float(info["no_com_progress_seconds"]),
                "recovery_grace_used": int(info["recovery_grace_used"]),
                "reach_1m": int(info["distance_m"] >= 1.0),
                "reach_5m": int(info["distance_m"] >= 5.0),
                "reach_10m": int(info["distance_m"] >= 10.0),
                "reach_100m": int(info["com_max_x_m"] >= 100.0),
            })
    finally:
        env.close()

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {
        "checkpoint": label,
        "steps": model.num_timesteps,
        "episodes": len(rows),
        "goal_distance_m": args.goal_distance,
        "mean_episode_speed_mps": mean("episode_speed_mps"),
        "mean_net_forward_m": mean("net_forward_m"),
        "mean_abs_lateral_m": mean("mean_abs_lateral_m"),
        "mean_max_lateral_m": mean("max_abs_lateral_m"),
        "mean_path_length_m": mean("path_length_m"),
        "mean_straightness": mean("straightness"),
        "mean_front_m": mean("front_m"),
        "mean_walking_m": mean("walking_m"),
        "mean_upright_pct": mean("upright_pct"),
        "mean_alternating_steps": mean("alternating_steps"),
        "two_steps_pct": 100.0 * float(np.mean([r["alternating_steps"] >= 2 for r in rows])),
        "mean_clf_value": mean("mean_clf_value"),
        "mean_clf_violation": mean("mean_clf_violation"),
        "mean_foot_slip": mean("mean_foot_slip"),
        "fallen_count": sum(r["reason"] == "fallen" for r in rows),
        "walk_1m_pct": 100.0 * float(np.mean([r["walking_m"] >= 1.0 for r in rows])),
        "walk_5m_pct": 100.0 * float(np.mean([r["walking_m"] >= 5.0 for r in rows])),
        "mean_com_m": mean("com_m"),
        "mean_rear_foot_m": mean("rear_foot_m"),
        "mean_episode_s": mean("seconds"),
        "stalled_count": sum(r["reason"] == "stalled" for r in rows),
        "time_limit_count": sum(r["reason"] == "time_limit" for r in rows),
        "goal_count": sum(r["reason"] == "goal" for r in rows),
        "reach_1m_pct": 100.0 * mean("reach_1m"),
        "reach_5m_pct": 100.0 * mean("reach_5m"),
        "reach_10m_pct": 100.0 * mean("reach_10m"),
        "reach_100m_pct": 100.0 * mean("reach_100m"),
        "mean_low_posture_pct": mean("low_posture_pct"),
        "mean_hand_contact_pct": mean("hand_contact_pct"),
    }

    def append_csv(path, data):
        present = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            if not present:
                writer.writeheader()
            writer.writerows(data)

    append_csv(out / "evaluation_summary.csv", [summary])
    append_csv(out / "evaluation_episodes.csv", rows)
    if video and not should_stop(args.stop_file):
        runner.OUT = out
        recording_args = SimpleNamespace(width=1.0, seconds=seconds,
                                        stall_seconds=args.stall_window)
        # Keep renderer chatter out of progress logs; errors still propagate.
        with contextlib.redirect_stdout(io.StringIO()):
            runner.record_video(model, recording_args, label, seed=5000)
    return summary


def atomic_save(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".saving.zip")
    model.save(str(temporary))
    os.replace(temporary, path)


def load_start_model(source, vec, args):
    """359-D standing models and 366-D locomotion models are selectable."""
    old = PPO.load(str(source), device="cpu")
    if old.action_space.shape != vec.action_space.shape or not (
            np.allclose(old.action_space.low, vec.action_space.low)
            and np.allclose(old.action_space.high, vec.action_space.high)):
        raise ValueError("開始モデルの行動空間が現在の物理モデルと一致しません")
    if old.observation_space.shape == vec.observation_space.shape:
        return PPO.load(str(source), env=vec, device="cpu", n_steps=args.rollout_total // args.envs,
                        learning_rate=3e-5, seed=args.seed, verbose=0), "direct_366"
    if old.observation_space.shape != (359,) or vec.observation_space.shape != (366,):
        raise ValueError(f"対応していない観測次元: {old.observation_space.shape}")
    kwargs = {k: getattr(old, k) for k in (
        "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "vf_coef",
        "max_grad_norm", "use_sde", "sde_sample_freq", "target_kl", "normalize_advantage")}
    kwargs["clip_range"] = float(old.clip_range(1.0))
    if old.clip_range_vf is not None:
        kwargs["clip_range_vf"] = float(old.clip_range_vf(1.0))
    model = PPO(old.policy_class, vec, learning_rate=3e-5,
                n_steps=args.rollout_total // args.envs,
                policy_kwargs=old.policy_kwargs, seed=args.seed, device="cpu", verbose=0, **kwargs)
    weights = model.policy.state_dict()
    expanded = []
    for key, src in old.policy.state_dict().items():
        dst = weights[key]
        if src.shape == dst.shape:
            dst.copy_(src)
        elif src.ndim == dst.ndim == 2 and src.shape[0] == dst.shape[0] and dst.shape[1] == 366 and src.shape[1] == 359:
            dst.zero_()
            dst[:, :359].copy_(src)
            expanded.append(key)
        else:
            raise ValueError(f"重みを引き継げません: {key}: {src.shape} -> {dst.shape}")
    if len(expanded) != 2:
        raise ValueError(f"Actor/Criticの入力拡張を確認できません: {expanded}")
    model.policy.load_state_dict(weights)
    # Verify identical deterministic actions for arbitrary extra task inputs.
    rng = np.random.default_rng(123)
    probe = rng.normal(0, .1, (8, 366)).astype(np.float32)
    with torch.no_grad():
        old_obs = torch.as_tensor(probe[:, :359], device=old.device)
        new_obs = torch.as_tensor(probe, device=model.device)
        old_mean = old.policy.get_distribution(old_obs).distribution.mean
        new_mean = model.policy.get_distribution(new_obs).distribution.mean
        if not torch.allclose(old_mean, new_mean, atol=1e-6, rtol=1e-5):
            raise ValueError("観測拡張前後でActor出力が一致しません")
    model.num_timesteps = old.num_timesteps
    return model, "standing_359_to_366_fresh_optimizer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-curve", choices=("log", "sigmoid"), default="log")
    parser.add_argument("--log-scale", type=float, default=0.25)
    parser.add_argument("--speed-weight", type=float, default=2.0)
    parser.add_argument("--fall-penalty", type=float, default=10.0)
    parser.add_argument("--time-cost", type=float, default=0.05)
    parser.add_argument("--eval-seed-start", type=int, default=6000)
    parser.add_argument("--posture-height-band", type=float, default=0.30,
                        help="高さ誤差に罰を入れない幅m")
    parser.add_argument("--posture-angle-band", type=float, default=30.0,
                        help="roll/pitch誤差に罰を入れない幅deg")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="このファイルを作ると保存して終了。既定は保存先/STOP")
    parser.add_argument("--goal-tolerance", type=float, default=0.30,
                        help="ゴール点からの許容距離m")
    parser.add_argument("--line-scale", type=float, default=0.75,
                        help="横ずれの評価尺度m。物理的な幅制約ではない")
    parser.add_argument("--line-weight", type=float, default=0.30)
    parser.add_argument("--decay-weight", type=float, default=0.15,
                        help="減衰条件違反の重み。0で追跡のみの比較")
    parser.add_argument("--height-drop", type=float, default=0.30,
                        help="開始胴体高から許す低下m")
    parser.add_argument("--min-upright", type=float, default=0.75,
                        help="胴体上方向と世界鉛直の内積下限")
    parser.add_argument("--goal-distance", type=float, default=10.0,
                        help="開始からのCOM前進による目標m")
    parser.add_argument("--fall-height-drop", type=float, default=0.55)
    parser.add_argument("--fall-alignment", type=float, default=0.17364817766693041,
                        help="明確な転倒: 鉛直から約80度の傾き")
    parser.add_argument("--fall-grace", type=float, default=0.50,
                        help="明確な転倒姿勢がこの秒数続いたら終了")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True,
                        help="専用の保存先。新規実験では空のフォルダーを指定")
    parser.add_argument("--resume", action="store_true",
                        help="指定保存先のmodel_latest.zipから追加学習")
    parser.add_argument("--rollout-total", type=int, default=2048,
                        help="PPOの1更新の総標本数。envsとbatch_sizeで割り切れる値")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="全環境合計の追加ステップ数")
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--stall-window", type=float, default=0.0, help="既定0で停滞終了なし")
    parser.add_argument("--stall-distance", type=float, default=0.10)
    parser.add_argument("--startup-grace", type=float, default=4.0)
    parser.add_argument("--source", "--model", dest="source", type=Path, default=None, help="開始モデル。省略時は速度版500万追加後モデル")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.envs <= 16:
        parser.error("--envs は1〜16")
    if (args.steps < 0 or args.seconds <= 0 or args.eval_every <= 0
        or args.episodes <= 0 or args.stall_window < 0
        or args.stall_distance <= 0 or args.startup_grace < 0):
        parser.error("各ステップ・時間・停滞判定の設定を確認してください")

    if not np.isfinite(args.log_scale) or args.log_scale <= 0 or not np.isfinite(args.decay_weight) or args.decay_weight < 0:
        parser.error("log-scaleは正、decay-weightは0以上にしてください")
    if args.height_drop <= 0 or not 0 < args.min_upright <= 1 or args.fall_grace <= 0:
        parser.error("姿勢判定の範囲を確認してください")
    if (not np.isfinite(args.goal_distance) or args.goal_distance <= 0
        or not np.isfinite(args.fall_height_drop) or args.fall_height_drop <= args.height_drop
        or not -1 <= args.fall_alignment < args.min_upright):
        parser.error("目標距離・転倒閾値を確認してください")
    if (not np.isfinite(args.line_scale) or args.line_scale <= 0
        or not np.isfinite(args.line_weight) or args.line_weight < 0
        or not np.isfinite(args.goal_tolerance) or args.goal_tolerance <= 0 or args.goal_tolerance >= args.goal_distance):
        parser.error("line-scaleは正、line-weightは0以上にしてください")
    if (not 0 <= args.posture_height_band < args.fall_height_drop
        or not 0 <= args.posture_angle_band < 80):
        parser.error("前進重み・姿勢許容幅を確認してください")
    if (any(not np.isfinite(v) or v < 0 for v in
            (args.speed_weight, args.fall_penalty, args.time_cost)) or args.eval_seed_start < 0):
        parser.error("速度重み・転倒罰・時間コスト・評価seedを確認してください")
    os.chdir(ROOT)
    torch.set_num_threads(1)

    out = args.out.resolve()
    latest = out / "model_latest.zip"
    args.stop_file = args.stop_file.resolve() if args.stop_file else out / "STOP"
    if args.resume and args.source is not None:
        parser.error("--resumeと--sourceは同時に指定できません")
    if args.rollout_total < args.envs or args.rollout_total % args.envs:
        parser.error("--rollout-totalはenvsで割り切れる正の値にしてください")
    if args.resume and not latest.is_file():
        parser.error(f"再開モデルがありません: {latest}")
    if not args.resume and out.exists() and any(out.iterdir()):
        parser.error("保存先が空ではありません。別の--outか、--resumeを指定してください")
    out.mkdir(parents=True, exist_ok=True)

    # Never silently fall back to a crawling/previous parallel model.
    source = args.source.resolve() if args.source is not None else STANDING_MODEL
    if args.resume:
        source = latest
    if not args.test and not source.is_file():
        parser.error(f"開始モデルがありません: {source}")

    if not GROUND_XML.is_file():
        import train_flat100_groundfix_camera as cfg
        cfg.solid_ground_xml(1.0)

    if args.test:
        # In the smoke test, trigger a quick guaranteed stall without training.
        args.stall_window = 0.4
        args.stall_distance = 1000.0
        args.startup_grace = 0.0
        args.fall_grace = 1000.0

    runner = configure_existing_env(out, args)
    n_steps = args.rollout_total // args.envs
    print(f"\n並列{args.envs}体 | 1体{n_steps} steps / PPO更新 | 合計{n_steps*args.envs}")
    print(f"評価/結果保存先: {out}")
    print(f"停滞終了: {'無効' if args.stall_window == 0 else str(args.stall_window) + '秒'} | COM目標: {args.goal_distance:g}m")
    print(f"開始猶予: {args.startup_grace:g}秒 / 転倒終了猶予: {args.fall_grace:g}秒（回復猶予は不使用）")

    worker_fns = [make_worker(i, str(out), args.seconds, args.seed, args)
                  for i in range(args.envs)]
    vec = (SubprocVecEnv(worker_fns, start_method="spawn")
           if args.envs > 1 else DummyVecEnv(worker_fns))

    try:
        vec.seed(args.seed)
        obs = vec.reset()
        print("観測形状:", obs.shape, "行動形状:", vec.action_space.shape)
        assert obs.shape == (args.envs, 366), obs.shape
        assert vec.action_space.shape == (17,), vec.action_space.shape

        if args.test:
            reasons = []
            for _ in range(80):
                obs, rew, done, infos = vec.step(
                    np.zeros((args.envs, 17), dtype=np.float32)
                )
                reasons.extend(i.get("reason") for i in infos if "episode" in i)
                if "stalled" in reasons:
                    break
            print("テスト終了理由:", reasons)
            assert "stalled" in reasons, "停滞終了が起きませんでした"
            print(f"★ {args.envs}体並列・停滞による自動リセットの動作確認OK")
            return

        if not source.is_file():
            raise FileNotFoundError(f"初期モデルがありません: {source}")
        model_file = source
        model, transfer_mode = load_start_model(model_file, vec, args)
        print(f"開始モデル: {model_file} | 累積 {model.num_timesteps:,} | {transfer_mode}", flush=True)
        if args.rollout_total % model.batch_size:
            raise ValueError(f"rollout-total={args.rollout_total} must divide into batch_size={model.batch_size}")
        if args.resume:
            manifest_path = out / "run_config.json"
            if not manifest_path.is_file():
                raise ValueError("run_config.jsonがないため設定を確認できません。--sourceと新しい--outで開始してください")
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("reward_version") != REWARD_VERSION:
                raise ValueError("報酬版が異なります。--sourceと新しい--outで開始してください")
            for key in ("envs", "rollout_total", "seconds", "stall_window", "stall_distance", "startup_grace", "seed", "height_drop", "min_upright", "fall_grace", "log_scale", "speed_curve", "decay_weight", "goal_distance", "fall_height_drop", "fall_alignment", "line_scale", "line_weight", "goal_tolerance", "posture_height_band", "posture_angle_band", "speed_weight", "fall_penalty", "time_cost", "eval_seed_start"):
                if old.get(key) != getattr(args, key):
                    raise ValueError(f"再開設定が異なります: {key}; 新しい--outと--sourceを指定してください")
        manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        manifest.update(transfer_mode=transfer_mode, reward_version=REWARD_VERSION, source_model=str(model_file), start_timesteps=model.num_timesteps,
                        n_steps=n_steps, batch_size=model.batch_size)
        (out / "run_config.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        # Install explicit handlers in parent after workers are spawned.
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        (out / "train.pid").write_text(str(os.getpid()), encoding="utf-8")
        reporter = EpisodeReporter(args.stop_file)
        # Starting measurement to compare against before the new rule.
        if not (out / "evaluation_summary.csv").exists():
            initial_summary = evaluate_and_record(runner, model, out, "initial", args.seconds,
                                args.episodes, args.video, args)
            if initial_summary is None or should_stop(args.stop_file):
                atomic_save(model, latest)
                print("停止要求を受信。モデルを保存して終了します。", flush=True)
                return
            print(f"初期評価{args.episodes}試行: COM平均={initial_summary['mean_com_m']:.2f}m 横ずれ平均={initial_summary['mean_abs_lateral_m']:.2f}m", flush=True)

        start_steps = model.num_timesteps
        training_seconds = 0.0
        started = time.perf_counter()
        final_target = start_steps + args.steps
        print(f"今回の追加目標: {args.steps:,}; 累積目標: {final_target:,} (rollout単位で超過する場合あり)")
        try:
            while model.num_timesteps < final_target:
                if should_stop(args.stop_file):
                    atomic_save(model, latest)
                    print("停止要求を受信。モデルを保存して終了します。", flush=True)
                    return
                report_index = (model.num_timesteps - start_steps) // args.eval_every + 1
                report_target = min(final_target, start_steps + report_index * args.eval_every)
                chunk = report_target - model.num_timesteps
                chunk_started = time.perf_counter()
                model.learn(total_timesteps=chunk,
                            reset_num_timesteps=False,
                            callback=reporter)
                training_seconds += time.perf_counter() - chunk_started
                if should_stop(args.stop_file):
                    atomic_save(model, latest)
                    print(f"停止要求を受信。累積{model.num_timesteps:,}のモデルを保存して終了します。", flush=True)
                    return
                label = f"step_{model.num_timesteps}"
                path = out / f"{label}.zip"
                atomic_save(model, path)
                atomic_save(model, latest)
                evaluation = evaluate_and_record(runner, model, out, label, args.seconds,
                                    args.episodes, args.video, args)
                if evaluation is None or should_stop(args.stop_file):
                    atomic_save(model, latest)
                    print("停止要求を受信。モデルを保存して終了します。", flush=True)
                    return
                elapsed = time.perf_counter() - started
                added = model.num_timesteps - start_steps
                timing = {"added_steps": added, "training_seconds": training_seconds,
                          "wall_seconds_excluding_initial_eval": elapsed,
                          "training_fps": added / training_seconds,
                          "wall_fps": added / elapsed}
                (out / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
                training_row = reporter.save_window(out, added, model.num_timesteps)
                train_mean = training_row['mean_com_max_x_m']
                train_text = f"{train_mean:.2f}m" if isinstance(train_mean, float) else "未完了"
                print(f"[追加{added/1e6:.2f}M / {args.steps/1e6:.2f}M] "
                      f"訓練COM平均={train_text}({training_row['episodes']}試行) | "
                      f"評価{args.episodes}試行: COM={evaluation['mean_com_m']:.2f}m "
                      f"速度={evaluation['mean_episode_speed_mps']:.3f}m/s 横ずれ={evaluation['mean_abs_lateral_m']:.2f}m "
                      f"直進率={evaluation['mean_straightness']:.0%} "
                      f"二歩={evaluation['two_steps_pct']:.0f}% "
                      f"転倒={evaluation['fallen_count']}/{args.episodes} 目標={evaluation['goal_count']}/{args.episodes} | "
                      f"{timing['wall_fps']:.0f}steps/s", flush=True)
        except KeyboardInterrupt:
            atomic_save(model, latest)
            print("中断時点のモデルを保存。再開は--sourceを外して--resumeを指定。--stepsはその再開時点からの追加量です")
            return
        except Exception:
            atomic_save(model, out / "model_error.zip")
            raise
        print("\n完了！ 保存先:", out)
    finally:
        vec.close()


if __name__ == "__main__":
    freeze_support()
    main()
