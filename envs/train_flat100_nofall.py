"""Phase A: unlimited flat ground, no off-beam or torso-fall termination.

Requirements: place alongside train_beam110.py, train_beam100.py,
train_flat_v5_sparse.py, etc., inside ~/rope.

Example:
  MUJOCO_GL=osmesa python -u train_flat100_nofall.py --test --seconds 40
  MUJOCO_GL=osmesa python -u train_flat100_nofall.py --init standing \
      --seconds 40 --steps 200000 --eval-every 50000 --episodes 20 --video
"""
import sys
sys.modules["triton"] = None  # work around CPU-only PPO.load crash

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

import train_beam110 as runner

# Keep all previous models / CSV / videos intact.
runner.OUT = Path("results/flat100_nofall")
runner.OUT.mkdir(parents=True, exist_ok=True)


def flat_xml(_width):
    """Replace the beam with a large flat plane at z=0.

    The floor's contact is physically infinite in MuJoCo (plane geom).
    World geoms named 'floor' and 'ground_below' are kept for compatibility
    with the existing standing and beam environment wrappers.
    """
    tree = ET.parse(runner.XML_SOURCE)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("XMLにworldbodyがありません")
    floor = world.find("./geom[@name='floor']")
    if floor is None:
        raise RuntimeError("元XMLにfloorがありません")
    floor.set("type", "plane")
    floor.set("pos", "0 0 0")
    floor.set("size", "160 30 0.1")
    floor.set("friction", "1.5 0.1 0.1")
    floor.set("condim", "3")
    floor.set("rgba", "0.65 0.71 0.70 1")

    # For the original beam wrapper's geom lookup only: no collision.
    # This dummy geom can never serve as a landing / failure surface.
    old_lower = world.find("./geom[@name='ground_below']")
    if old_lower is not None:
        world.remove(old_lower)
    ET.SubElement(world, "geom", {
        "name": "ground_below", "type": "plane",
        "pos": "0 0 -1000", "size": "1 1 0.1",
        "contype": "0", "conaffinity": "0", "rgba": "0 0 0 0",
    })

    # Visual-only reference marks. They are NOT raised obstacles.
    for x in range(0, 101, 5):
        c = "0.10 0.75 0.15 1" if x == 100 else (
            "0.18 0.45 0.95 1" if x == 0 else "0.93 0.82 0.30 1"
        )
        ET.SubElement(world, "geom", {
            "name": f"distance_mark_{x}m", "type": "box",
            "pos": f"{x} 0 0.002", "size": "0.035 1.0 0.001",
            "contype": "0", "conaffinity": "0", "rgba": c,
        })

    path = (runner.OUT / "flat_ground_100m.xml").resolve()
    tree.write(path, encoding="unicode")
    return path


class Flat100Env(runner.Beam110Env):
    """Reward forward movement; a fallen robot may crawl or stand back up."""

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.initial_front = self.front_x()
        self.initial_com = self.com_x()
        self.initial_rear = self.rear_foot_x()
        self.best_front = 0.0
        self.best_com = 0.0
        self.best_rear = 0.0
        self.next_cp_index = 1
        self.front_points = 0.0
        self.com_points = 0.0
        self.rear_points = 0.0
        self.cp_points = 0.0
        self.low_steps = 0
        self.total_steps = 0
        self.hands_on_floor_steps = 0
        return obs, info

    def com_x(self):
        return float(self.base.data.subtree_com[self.torso_id, 0])

    def rear_foot_x(self):
        d = self.base.data
        return min(float(d.geom("left_foot").xpos[0]),
                   float(d.geom("right_foot").xpos[0]))

    def step(self, action):
        # Physical simulation is still the same, including gravity and contact.
        # Discard the previous environment's fall/termination signals AND reward.
        raw_obs, _, _, _, info = self.base.step(action)
        info = dict(info)
        previous_reason = str(info.get("reason", ""))
        dt = float(self.base.dt)
        floor_contact, _, hand_contact = self.contacts()

        # All distances start at exactly zero, independent of limb lengths.
        current_front = max(0.0, self.front_x() - self.initial_front)
        current_com = max(0.0, self.com_x() - self.initial_com)
        current_rear = max(0.0, self.rear_foot_x() - self.initial_rear)

        # One-time reward for NEW forward records: shaking earns no repeats.
        new_front = min(runner.GOAL_X, max(self.best_front, current_front))
        new_com = min(runner.GOAL_X, max(self.best_com, current_com))
        new_rear = min(runner.GOAL_X, max(self.best_rear, current_rear))
        r_front = 2.0 * (new_front - self.best_front)
        r_com = 10.0 * (new_com - self.best_com)
        r_rear = 4.0 * (new_rear - self.best_rear)
        self.best_front, self.best_com, self.best_rear = (
            new_front, new_com, new_rear
        )

        # The 5m checkpoints still exist, but NEVER terminate the episode.
        r_cp = 0.0
        while (self.next_cp_index <= len(runner.CHECKPOINTS)
               and self.best_front >= runner.CHECKPOINTS[self.next_cp_index - 1]):
            cp = runner.CHECKPOINTS[self.next_cp_index - 1]
            r_cp += (cp / 10.0) * 5.0
            self.next_cp_index += 1

        self.front_points += r_front
        self.com_points += r_com
        self.rear_points += r_rear
        self.cp_points += r_cp

        # No off-beam, no falling, no stalled termination or penalty.
        # Small per-second time cost prioritizes making progress.
        reward = r_front + r_com + r_rear + r_cp - 0.02 * dt
        terminated = False
        truncated = False
        reason = "continuing"
        success = False

        self.total_steps += 1
        torso_z = float(self.base.data.body("torso").xpos[2])
        self.low_steps += int(torso_z < self.base.initial_z - 0.35)
        self.hands_on_floor_steps += int(hand_contact)

        if self.best_front >= runner.GOAL_X:
            reward += 300.0
            terminated = True
            reason = "goal"
            success = True
        elif float(self.base.elapsed) >= self.seconds:
            truncated = True
            reason = "time_limit"

        info.update({
            "reason": reason,
            "old_reason": previous_reason,
            "is_success": success,
            "distance_m": float(self.best_front),
            "front_x_m": float(self.front_x() - self.initial_front),
            "com_max_x_m": float(self.best_com),
            "rear_foot_max_x_m": float(self.best_rear),
            "checkpoints_passed": self.next_cp_index - 1,
            "beam_contact": bool(floor_contact),  # flat ground contact
            "hand_contact": bool(hand_contact),
            "low_posture_pct": 100.0 * self.low_steps / self.total_steps,
            "hand_contact_pct": 100.0 * self.hands_on_floor_steps / self.total_steps,
            "front_points_total": float(self.front_points),
            "com_points_total": float(self.com_points),
            "rear_points_total": float(self.rear_points),
            "checkpoint_points_total": float(self.cp_points),
        })
        return self.augment(raw_obs), float(reward), terminated, truncated, info


# Reuse the existing validated PPO warm-start, 366-D observation, eval,
# checkpoint and recording logic. Keep output separate from all beam runs.
runner.beam_xml = flat_xml
runner.Beam110Env = Flat100Env


def evaluate_ground(model, args, label):
    env = runner.make_env(args.width, args.seconds, args.stall_seconds)
    rows = []
    try:
        for i in range(args.episodes):
            obs, _ = env.reset(seed=3000 + i)
            while True:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(a)
                if done or trunc:
                    break
            rows.append({
                "label": label,
                "steps": model.num_timesteps,
                "seed": 3000 + i,
                "front_m": float(info["distance_m"]),
                "com_m": float(info["com_max_x_m"]),
                "rear_foot_m": float(info["rear_foot_max_x_m"]),
                "seconds": float(info["elapsed_seconds"]),
                "reason": info["reason"],
                "low_posture_pct": float(info["low_posture_pct"]),
                "hand_contact_pct": float(info["hand_contact_pct"]),
                "reward_front": float(info["front_points_total"]),
                "reward_com": float(info["com_points_total"]),
                "reward_rear": float(info["rear_points_total"]),
                "reward_cp": float(info["checkpoint_points_total"]),
                **{f"reach_{mark}m": int(info["distance_m"] >= mark)
                   for mark in runner.REACH_MARKS},
            })
    finally:
        env.close()

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "label": label,
        "steps": model.num_timesteps,
        "episodes": len(rows),
        "mean_front_m": mean("front_m"),
        "max_front_m": max(r["front_m"] for r in rows),
        "mean_com_m": mean("com_m"),
        "mean_rear_foot_m": mean("rear_foot_m"),
        "mean_seconds": mean("seconds"),
        "mean_low_posture_pct": mean("low_posture_pct"),
        "mean_hand_contact_pct": mean("hand_contact_pct"),
        "goal_count": sum(r["reason"] == "goal" for r in rows),
        "time_limit_count": sum(r["reason"] == "time_limit" for r in rows),
        **{f"reach_{m}m_pct": 100.0 * mean(f"reach_{m}m")
           for m in runner.REACH_MARKS},
    }
    runner.append_csv(runner.OUT / "evaluation_episodes.csv", rows)
    runner.append_csv(runner.OUT / "evaluation_summary.csv", [summary])
    print(f"\n[{label}] Steps {model.num_timesteps:,} | 地面の移動能力")
    print(f"  先端={summary['mean_front_m']:.3f}m、"
          f"重心={summary['mean_com_m']:.3f}m、"
          f"後足={summary['mean_rear_foot_m']:.3f}m")
    print(f"  1m={summary['reach_1m_pct']:.0f}%、"
          f"5m={summary['reach_5m_pct']:.0f}%、"
          f"10m={summary['reach_10m_pct']:.0f}%、"
          f"100m={summary['reach_100m_pct']:.0f}%")
    print(f"  低姿勢={summary['mean_low_posture_pct']:.1f}%、"
          f"手の接地={summary['mean_hand_contact_pct']:.1f}%")
    print(f"  時間切れ={summary['time_limit_count']}/{len(rows)}、"
          f"ゴール={summary['goal_count']}/{len(rows)}、落下終了=0（廃止）")
    return summary


runner.evaluate = evaluate_ground

if __name__ == "__main__":
    # runner's CLI still says "beam" in a couple of status messages;
    # actual XML is unlimited flat ground. "--width 1.0" is only a retained
    # input-feature scale for compatibility; there is NO physical boundary.
    print("【平地・落下なし学習】広い地面で自由に移動／転倒継続", flush=True)
    runner.main()
