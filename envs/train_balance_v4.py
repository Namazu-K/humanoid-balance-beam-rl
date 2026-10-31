import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from train_stand_endurance import StandEnduranceEnv


OUT = Path("results")
OUT.mkdir(exist_ok=True)

START = OUT / "phase0_endurance_1500k.zip"
BEST = OUT / "phase0_v4_best"
CSV_FILE = OUT / "phase0_v4_evaluation.csv"

# 累計ステップ、平行移動速度、角速度、初期傾斜
STAGES = [
    (1_800_000, 0.00, 0.00, 0.0),
    (2_200_000, 0.04, 0.08, 0.5),
    (2_600_000, 0.08, 0.15, 1.0),
    (3_000_000, 0.12, 0.20, 1.5),
]


class DynamicBalanceEnv(StandEnduranceEnv):
    """
    Phase 0 Ver.4: 動的バランス

    1. 胴体の水平移動ペナルティを除去
    2. 重心を接地足の近くに保つ報酬を追加
    3. 初期姿勢・速度の小さな乱れを追加
    4. 1秒で終了しない
    """

    def __init__(
        self,
        episode_limit=20.0,
        push_v=0.0,
        push_w=0.0,
        tilt_deg=0.0,
    ):
        super().__init__(
            episode_limit=episode_limit
        )

        self.push_v = float(push_v)
        self.push_w = float(push_w)
        self.tilt_deg = float(tilt_deg)

        self.torso_body_id = (
            self.model.body("torso").id
        )

    def set_disturbance(
        self, push_v, push_w, tilt_deg
    ):
        self.push_v = float(push_v)
        self.push_w = float(push_w)
        self.tilt_deg = float(tilt_deg)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(
            seed=seed, options=options
        )

        if (
            self.push_v == 0.0
            and self.push_w == 0.0
            and self.tilt_deg == 0.0
        ):
            return obs, info

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        rng = self.np_random

        # 水平方向の初速度
        qvel[0:2] += rng.uniform(
            -self.push_v,
            self.push_v,
            size=2
        )

        # 胴体の回転方向の初速度
        qvel[3:5] += rng.uniform(
            -self.push_w,
            self.push_w,
            size=2
        )

        # 初期の小さな前後左右の傾き
        if self.tilt_deg > 0:
            roll, pitch = np.deg2rad(
                rng.uniform(
                    -self.tilt_deg,
                    self.tilt_deg,
                    size=2
                )
            )

            qx = np.array([
                np.cos(roll / 2),
                np.sin(roll / 2),
                0.0,
                0.0
            ])

            qy = np.array([
                np.cos(pitch / 2),
                0.0,
                np.sin(pitch / 2),
                0.0
            ])

            combined = np.zeros(4)
            rotated = np.zeros(4)

            mujoco.mju_mulQuat(
                combined, qy, qx
            )
            mujoco.mju_mulQuat(
                rotated,
                combined,
                qpos[3:7].copy()
            )

            qpos[3:7] = (
                rotated / np.linalg.norm(rotated)
            )

        # MuJoCoの状態を更新
        self.physics.set_state(qpos, qvel)

        # 初期姿勢を更新後の状態に合わせる
        torso = self.data.body("torso")

        self.initial_z = float(torso.xpos[2])
        self.initial_xy = (
            np.asarray(torso.xpos[:2]).copy()
        )

        z, angle, posture, height, contact = (
            self._status()
        )

        # 観測359次元を維持する
        raw_obs = self.physics._get_obs()
        obs = self._observation(
            raw_obs, posture, contact
        )

        return obs, info

    def _support_quality(self):
        """
        全身重心の水平位置と、接地足の中心の距離。
        真の支持多角形ではなく近似的な指標。
        """

        contacting_feet = set()

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a = int(c.geom1)
            b = int(c.geom2)

            if a == self.floor_id and b in self.foot_ids:
                contacting_feet.add(b)

            if b == self.floor_id and a in self.foot_ids:
                contacting_feet.add(a)

        if not contacting_feet:
            return 0.0, 0.0

        feet_xy = np.array([
            self.data.geom_xpos[g, :2]
            for g in sorted(contacting_feet)
        ])

        support_center = feet_xy.mean(axis=0)

        # torsoを根とするロボット全身の重心
        com_xy = np.asarray(
            self.data.subtree_com[
                self.torso_body_id, :2
            ]
        )

        distance = float(
            np.linalg.norm(
                com_xy - support_center
            )
        )

        # 距離が小さいほど1に近い
        quality = float(
            np.exp(
                -0.5 * (distance / 0.30) ** 2
            )
        )

        return quality, distance

    def step(self, action):
        # Ver.3の物理計算、転倒判定、
        # 継続直立報酬を引き継ぐ
        obs, reward, done, truncated, info = (
            super().step(action)
        )

        torso_xy = np.asarray(
            self.data.body("torso").xpos[:2]
        )

        drift = float(
            np.linalg.norm(
                torso_xy - self.initial_xy
            )
        )

        support_q, support_distance = (
            self._support_quality()
        )

        # Ver.3にあった -0.5*drift を打ち消す
        reward += self.dt * 0.5 * drift

        # 重心が接地足の近くにあり、
        # 胴体が直立していると報酬
        reward += self.dt * (
            1.5
            * support_q
            * info["posture_quality"]
            * info["height_quality"]
        )

        info["support_quality"] = support_q
        info["support_distance_m"] = (
            support_distance
        )

        return (
            obs,
            float(reward),
            done,
            truncated,
            info
        )


class ProgressReport(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = 0
        self.recent = deque(maxlen=100)

    def _on_step(self):
        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", [])
        ):
            if not done:
                continue

            self.episodes += 1
            t = float(
                info.get(
                    "longest_stable_seconds", 0.0
                )
            )
            self.recent.append(t)

            if self.episodes % 100 == 0:
                a = np.asarray(self.recent)

                print(
                    f"\nStep={self.num_timesteps:,}"
                    f" | Episode={self.episodes}"
                )
                print(
                    f"直近平均安定: {a.mean():.3f}秒"
                )
                print(
                    f"1秒以上: {np.mean(a >= 1)*100:.1f}%"
                )
                print(
                    f"2秒以上: {np.mean(a >= 2)*100:.1f}%"
                )
                print(
                    f"3秒以上: {np.mean(a >= 3)*100:.1f}%"
                )

        return True


def evaluate(
    model,
    push_v=0.0,
    push_w=0.0,
    tilt_deg=0.0,
    episodes=100,
):
    env = DynamicBalanceEnv(
        episode_limit=20.0,
        push_v=push_v,
        push_w=push_w,
        tilt_deg=tilt_deg,
    )

    times = []
    survival = []

    try:
        for seed in range(1000, 1000 + episodes):
            obs, _ = env.reset(seed=seed)

            while True:
                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                obs, reward, done, trunc, info = (
                    env.step(action)
                )

                if done or trunc:
                    times.append(
                        info["longest_stable_seconds"]
                    )
                    survival.append(
                        info["elapsed_seconds"]
                    )
                    break
    finally:
        env.close()

    a = np.asarray(times)

    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "survival": float(
            np.mean(survival)
        ),
        "over_1s": int(np.sum(a >= 1.0)),
        "over_2s": int(np.sum(a >= 2.0)),
        "over_3s": int(np.sum(a >= 3.0)),
        "over_5s": int(np.sum(a >= 5.0)),
        "over_10s": int(np.sum(a >= 10.0)),
    }


def evaluate_both(model):
    # 従来と同じ初期条件
    normal = evaluate(model)

    # 共通の軽い外乱を与えた条件
    disturbed = evaluate(
        model,
        push_v=0.08,
        push_w=0.15,
        tilt_deg=1.0,
    )

    return normal, disturbed


def display(label, normal, disturbed):
    print("\n" + "=" * 58)
    print(label)
    print("=" * 58)

    for title, s in [
        ("通常条件", normal),
        ("外乱あり", disturbed)
    ]:
        print(f"\n{title}")
        print(
            f"平均安定時間: {s['mean']:.3f}秒"
        )
        print(
            f"中央値: {s['median']:.3f}秒"
        )
        print(
            f"最高記録: {s['max']:.3f}秒"
        )
        print(
            f"1秒以上: {s['over_1s']}/100"
        )
        print(
            f"2秒以上: {s['over_2s']}/100"
        )
        print(
            f"3秒以上: {s['over_3s']}/100"
        )
        print(
            f"5秒以上: {s['over_5s']}/100"
        )


def save_csv(rows):
    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def score_of(normal, disturbed):
    # 通常と外乱ありの平均安定時間を同等に評価
    return (
        normal["mean"] + disturbed["mean"]
    ) / 2.0


def smoke_test():
    env = DynamicBalanceEnv(
        push_v=0.08,
        push_w=0.15,
        tilt_deg=1.0
    )

    try:
        obs, info = env.reset(seed=42)

        print("観測次元:", obs.shape)
        print("行動次元:", env.action_space.shape)

        assert obs.shape == (359,)

        for i in range(1500):
            action = np.zeros(
                env.action_space.shape,
                dtype=np.float32
            )

            obs, reward, done, trunc, info = (
                env.step(action)
            )

            if done or trunc:
                print("終了理由:", info["reason"])
                print(
                    "最長安定:",
                    f"{info['longest_stable_seconds']:.3f}秒"
                )
                print(
                    "重心と接地足の距離:",
                    f"{info['support_distance_m']:.3f}m"
                )
                break

        print("動作テスト完了")

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true"
    )
    parser.add_argument(
        "--model",
        default=str(START)
    )
    args = parser.parse_args()

    if args.test:
        smoke_test()
        return

    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"開始モデルがありません: {args.model}"
        )

    train_env = DynamicBalanceEnv()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    env = Monitor(
        train_env,
        filename=str(
            OUT / f"phase0_v4_{stamp}.monitor.csv"
        ),
        info_keywords=(
            "reason",
            "longest_stable_seconds",
        ),
    )

    model = PPO.load(
        args.model,
        env=env,
        device="cpu",
        learning_rate=5e-5,
        gamma=0.999,
        target_kl=0.015,
    )

    print("\n=== Phase 0 Ver.4 動的バランス ===")
    print("開始モデル:", args.model)
    print("累計ステップ:", model.num_timesteps)
    print("観測次元:", env.observation_space.shape)
    print("行動次元:", env.action_space.shape)
    print("学習率:", model.learning_rate)

    rows = []
    completed = False
    reporter = ProgressReport()

    try:
        normal, disturbed = evaluate_both(model)
        display(
            "学習開始前",
            normal,
            disturbed
        )

        best_score = score_of(
            normal, disturbed
        )

        model.save(str(BEST))

        def append_record(target, n, d):
            rows.append({
                "target": target,
                "actual_steps": model.num_timesteps,
                "normal_mean": n["mean"],
                "normal_max": n["max"],
                "normal_over_2s": n["over_2s"],
                "normal_over_3s": n["over_3s"],
                "disturbed_mean": d["mean"],
                "disturbed_max": d["max"],
                "disturbed_over_2s": d["over_2s"],
                "disturbed_over_3s": d["over_3s"],
                "combined_score": score_of(n, d),
            })
            save_csv(rows)

        append_record(
            model.num_timesteps,
            normal,
            disturbed
        )

        for target, pv, pw, tilt in STAGES:
            remaining = (
                target - model.num_timesteps
            )

            if remaining <= 0:
                continue

            train_env.set_disturbance(
                pv, pw, tilt
            )

            print("\n" + "#" * 58)
            print(
                f"目標: {target:,} Step"
            )
            print(
                f"初速度乱れ: ±{pv:.2f}m/s"
            )
            print(
                f"角速度乱れ: ±{pw:.2f}rad/s"
            )
            print(
                f"姿勢乱れ: ±{tilt:.1f}度"
            )
            print("#" * 58)

            model.learn(
                total_timesteps=remaining,
                reset_num_timesteps=False,
                callback=reporter
            )

            label = target // 1000
            path = (
                OUT / f"phase0_v4_{label}k"
            )

            model.save(str(path))

            normal, disturbed = evaluate_both(
                model
            )

            display(
                f"累計 {model.num_timesteps:,} Step",
                normal,
                disturbed
            )

            append_record(
                target,
                normal,
                disturbed
            )

            current_score = score_of(
                normal, disturbed
            )

            if current_score > best_score:
                best_score = current_score
                model.save(str(BEST))
                print(
                    "★ 最良モデル更新:",
                    str(BEST) + ".zip"
                )

        completed = True

        print("\n=== Ver.4 学習完了 ===")

        for row in rows:
            print(
                f"{row['actual_steps']:,} Step"
                f" | 通常={row['normal_mean']:.3f}秒"
                f" | 外乱={row['disturbed_mean']:.3f}秒"
                f" | 3秒達成={row['normal_over_3s']}/100"
            )

        print(
            "\n最良モデル:",
            str(BEST) + ".zip"
        )
        print("結果CSV:", CSV_FILE)

    except KeyboardInterrupt:
        print("\n手動で中断しました。")

    finally:
        if not completed:
            backup = OUT / "phase0_v4_interrupted"
            model.save(str(backup))
            print(
                "途中モデルを保存:",
                str(backup) + ".zip"
            )

        env.close()


if __name__ == "__main__":
    main()
