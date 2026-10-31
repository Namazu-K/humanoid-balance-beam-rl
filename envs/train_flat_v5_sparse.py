import sys

# CPU実行時のTriton読み込みクラッシュ対策
sys.modules["triton"] = None

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

import stand_env
from train_balance_v4 import DynamicBalanceEnv
from train_flat_scratch import (
    FLAT_XML,
    ORIGINAL_GYM_MAKE,
)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ==========================================
# Ver.5 新しい報酬設計
# ==========================================

class FlatBalanceV5(DynamicBalanceEnv):

    def __init__(self):
        super().__init__(
            episode_limit=20.0,
            push_v=0.0,
            push_w=0.0,
            tilt_deg=0.0,
        )

        self.relaxed_streak = 0.0
        self.longest_relaxed = 0.0
        self.prev_action = np.zeros(
            self.action_space.shape,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        self.relaxed_streak = 0.0
        self.longest_relaxed = 0.0
        self.prev_action = np.zeros(
            self.action_space.shape,
            dtype=np.float32,
        )

        return obs, info

    def step(self, action):

        # 物理計算と転倒判定はVer.4を継承
        obs, old_reward, done, trunc, info = (
            super().step(action)
        )

        z = float(info["torso_height_m"])
        angle = float(info["torso_angle_deg"])

        posture = float(
            info["posture_quality"]
        )

        contact = bool(
            info["feet_contact"]
        )

        support = float(
            info["support_quality"]
        )

        # ----------------------------------
        # 変更1：しゃがみ姿勢を許容
        # ----------------------------------

        # 従来は20cmの下降で安定判定から外れた。
        # 新報酬では38cmまでの下降を許容。
        height = float(np.clip(
            (
                z - (self.initial_z - 0.55)
            ) / 0.17,
            0.0,
            1.0,
        ))

        # ----------------------------------
        # 変更2：多少の傾きも許容
        # ----------------------------------

        relaxed = bool(
            angle <= 45.0
            and z >= self.initial_z - 0.38
            and contact
        )

        if relaxed:
            self.relaxed_streak += self.dt

            self.longest_relaxed = max(
                self.longest_relaxed,
                self.relaxed_streak,
            )
        else:
            self.relaxed_streak = 0.0

        # ----------------------------------
        # 変更3：操作コスト
        # ----------------------------------

        action = np.asarray(
            action, dtype=np.float32
        )

        effort = float(
            np.mean((action / 0.4) ** 2)
        )


        # 関節の操作量を最大指令で正規化
        u = np.clip(
            action.astype(np.float64) / 0.4,
            -1.0, 1.0
        )
        prev_u = self.prev_action / 0.4

        # L1: 少ないモーター操作を優遇
        l1_effort = float(
            np.mean(np.abs(u))
        )

        # 急激な操作変更を軽く減点
        action_change = float(
            np.mean(np.abs(u - prev_u))
        )

        # 強く活動するモーターの実効個数
        soft_active = np.clip(
            (np.abs(u) - 0.20) / 0.30,
            0.0, 1.0
        )
        active_equivalent = float(
            np.sum(soft_active)
        )

        # 7個相当を超える強い同時操作だけ減点
        excessive = max(
            0.0,
            (active_equivalent - 7.0) / 10.0
        )
        multi_active_penalty = excessive ** 2

        self.prev_action = action.copy()

        angular_speed = float(
            np.linalg.norm(
                self.data.qvel[3:6]
            )
        )

        # ----------------------------------
        # Ver.5 報酬
        # ----------------------------------

        # 足で支持し、姿勢を保つほど得点。
        # しゃがみ・踏み替え自体は減点しない。
        # 直立時間の達成ボーナスもなし。

        reward_per_second = (
            0.4
            + 2.0 * posture * height * float(contact)
            + 1.4 * support * posture * height
            + 1.5 * float(relaxed)
            - 0.03 * effort
            - 0.04 * angular_speed
            - 0.05 * l1_effort
            - 0.02 * action_change
            - 0.06 * multi_active_penalty
        )

        reward = self.dt * reward_per_second

        if info["reason"] == "fall":
            reward -= 5.0

        # ----------------------------------
        # 計測情報
        # ----------------------------------

        # 旧基準の安定時間は
        # info["longest_stable_seconds"] に残す。
        info["longest_relaxed_seconds"] = float(
            self.longest_relaxed
        )

        info["relaxed_stable"] = relaxed
        info["new_height_quality"] = height

        info["active_actuators"] = int(
            np.sum(np.abs(u) >= 0.25)
        )
        info["action_change"] = action_change
        info["control_l1"] = l1_effort
        info["active_equivalent"] = active_equivalent

        return (
            obs,
            float(reward),
            done,
            trunc,
            info,
        )


# ==========================================
# 平足XMLの適用
# ==========================================

def make_env():

    def custom_make(*args, **kwargs):
        if args and args[0] == "Humanoid-v5":
            kwargs["xml_file"] = str(
                FLAT_XML
            )

        return ORIGINAL_GYM_MAKE(
            *args, **kwargs
        )

    with patch.object(
        stand_env.gym,
        "make",
        side_effect=custom_make,
    ):
        env = FlatBalanceV5()

    return env


# ==========================================
# 学習中の表示
# ==========================================

class Report(BaseCallback):

    def __init__(self):
        super().__init__()
        self.episodes = 0
        self.recent = deque(maxlen=100)

    def _on_step(self):

        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", []),
        ):
            if not done:
                continue

            self.episodes += 1

            self.recent.append(
                float(
                    info.get(
                        "longest_relaxed_seconds",
                        0.0,
                    )
                )
            )

            if self.episodes % 100 == 0:
                a = np.asarray(self.recent)

                print(
                    f"\n[Ver.5] "
                    f"Step={self.num_timesteps:,} "
                    f"Episode={self.episodes}",
                    flush=True,
                )
                print(
                    f"新基準平均安定: {a.mean():.3f}秒"
                )
                print(
                    f"2秒以上: {np.mean(a >= 2)*100:.1f}%"
                )
                print(
                    f"3秒以上: {np.mean(a >= 3)*100:.1f}%"
                )
                print(
                    f"5秒以上: {np.mean(a >= 5)*100:.1f}%"
                )

        return True


# ==========================================
# 評価
# ==========================================

def evaluate(model, episodes=100):

    env = make_env()

    strict_times = []
    relaxed_times = []
    survival_times = []
    knee_ranges = []
    active_counts = []
    action_changes = []

    try:
        # 膝関節の位置も計測する
        knee_addresses = []

        for name in (
            "left_knee",
            "right_knee",
        ):
            jid = env.model.joint(name).id
            knee_addresses.append(
                int(env.model.jnt_qposadr[jid])
            )

        for seed in range(
            3000, 3000 + episodes
        ):
            obs, _ = env.reset(seed=seed)

            knee_min = np.array([
                env.data.qpos[i]
                for i in knee_addresses
            ])

            knee_max = knee_min.copy()

            while True:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, done, trunc, info = (
                    env.step(action)
                )

                active_counts.append(
                    info["active_actuators"]
                )
                action_changes.append(
                    info["action_change"]
                )

                knees = np.array([
                    env.data.qpos[i]
                    for i in knee_addresses
                ])

                knee_min = np.minimum(
                    knee_min, knees
                )
                knee_max = np.maximum(
                    knee_max, knees
                )

                if done or trunc:
                    strict_times.append(
                        info["longest_stable_seconds"]
                    )

                    relaxed_times.append(
                        info["longest_relaxed_seconds"]
                    )

                    survival_times.append(
                        info["elapsed_seconds"]
                    )

                    knee_ranges.append(
                        np.degrees(
                            np.mean(
                                knee_max - knee_min
                            )
                        )
                    )

                    break

    finally:
        env.close()

    strict = np.asarray(strict_times)
    relaxed = np.asarray(relaxed_times)

    return {
        "strict_mean": float(strict.mean()),
        "relaxed_mean": float(relaxed.mean()),
        "survival_mean": float(
            np.mean(survival_times)
        ),
        "max_relaxed": float(relaxed.max()),
        "over_2s": int(np.sum(relaxed >= 2)),
        "over_3s": int(np.sum(relaxed >= 3)),
        "over_5s": int(np.sum(relaxed >= 5)),
        "knee_range_deg": float(
            np.mean(knee_ranges)
        ),
        "mean_active": float(np.mean(active_counts)),
        "mean_action_change": float(
            np.mean(action_changes)
        ),
    }


def show(label, r):
    print("\n" + "=" * 55)
    print(label)
    print("=" * 55)

    print(
        f"旧基準・平均安定: "
        f"{r['strict_mean']:.3f}秒"
    )
    print(
        f"新基準・平均安定: "
        f"{r['relaxed_mean']:.3f}秒"
    )
    print(
        f"平均生存時間: "
        f"{r['survival_mean']:.3f}秒"
    )
    print(
        f"新基準・最高記録: "
        f"{r['max_relaxed']:.3f}秒"
    )
    print(f"2秒以上: {r['over_2s']}/100")
    print(f"3秒以上: {r['over_3s']}/100")
    print(f"5秒以上: {r['over_5s']}/100")

    print(
        f"膝の平均可動範囲: "
        f"{r['knee_range_deg']:.1f}度"
    )

    print(
        f"平均活動アクチュエータ数: "
        f"{r['mean_active']:.2f}/17"
    )
    print(
        f"平均操作変化量: "
        f"{r['mean_action_change']:.4f}"
    )


def save_csv(rows, path):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 実行
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
    )

    parser.add_argument(
        "--mode",
        choices=["continue", "scratch"],
        default="continue",
    )

    args = parser.parse_args()

    if not FLAT_XML.exists():
        raise FileNotFoundError(
            f"平足XMLがありません: {FLAT_XML}"
        )

    if args.test:
        env = make_env()

        try:
            obs, _ = env.reset(seed=42)

            print("観測:", obs.shape)
            print("行動:", env.action_space.shape)

            assert obs.shape == (359,)
            assert env.action_space.shape == (17,)

            for _ in range(1500):
                action = np.zeros(
                    env.action_space.shape,
                    dtype=np.float32,
                )

                obs, reward, done, trunc, info = (
                    env.step(action)
                )

                if done or trunc:
                    print(
                        "旧基準安定:",
                        info["longest_stable_seconds"],
                    )
                    print(
                        "新基準安定:",
                        info["longest_relaxed_seconds"],
                    )
                    print("終了:", info["reason"])
                    break

            print("動作テスト成功")

        finally:
            env.close()

        return

    mode = args.mode

    prefix = (
        "flat_v5_sparse_ft"
        if mode == "continue"
        else "flat_v5_sparse_scratch"
    )

    if mode == "continue":
        targets = [
            2_200_000,
            2_500_000,
            3_000_000,
        ]
    else:
        targets = [
            200_000,
            500_000,
            1_000_000,
            1_500_000,
            2_000_000,
        ]

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    env = Monitor(
        make_env(),
        filename=str(
            RESULTS / f"{prefix}_{stamp}.monitor.csv"
        ),
        info_keywords=(
            "reason",
            "longest_stable_seconds",
            "longest_relaxed_seconds",
        ),
    )

    if mode == "continue":

        start = RESULTS / "flat_scratch_best.zip"

        if not start.exists():
            raise FileNotFoundError(start)

        print("既存モデルから追加学習:", start)

        model = PPO.load(
            str(start),
            env=env,
            device="cpu",
            learning_rate=5e-5,
            gamma=0.999,
            target_kl=0.02,
        )

    else:

        print("ランダム初期化から学習")

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.15,
            target_kl=0.02,
            ent_coef=0.0,
            policy_kwargs={
                "log_std_init": -2.0
            },
            seed=42,
            verbose=1,
            device="cpu",
        )

    best_path = RESULTS / f"{prefix}_best"
    csv_path = RESULTS / f"{prefix}_evaluation.csv"

    reporter = Report()

    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path=str(
            RESULTS / f"{prefix}_checkpoints"
        ),
        name_prefix=prefix,
    )

    callbacks = CallbackList([
        reporter,
        checkpoint,
    ])

    rows = []
    completed = False

    try:
        if mode == "continue":

            baseline = evaluate(model)
            show("開始時点・旧200万モデル", baseline)

            rows.append({
                "target": model.num_timesteps,
                "actual_steps": model.num_timesteps,
                **baseline,
            })

            best_score = baseline["relaxed_mean"]
            model.save(str(best_path))

        else:
            best_score = -float("inf")

        if rows:
            save_csv(rows, csv_path)

        for target in targets:

            remaining = target - model.num_timesteps

            if remaining <= 0:
                continue

            print(
                f"\n次の目標: {target:,}ステップ"
            )

            model.learn(
                total_timesteps=remaining,
                reset_num_timesteps=False,
                callback=callbacks,
            )

            path = (
                RESULTS
                / f"{prefix}_{target//1000}k"
            )

            model.save(str(path))

            result = evaluate(model)

            show(
                f"累計 {model.num_timesteps:,} Step",
                result,
            )

            rows.append({
                "target": target,
                "actual_steps": model.num_timesteps,
                **result,
            })

            save_csv(rows, csv_path)

            if result["relaxed_mean"] > best_score:

                best_score = result["relaxed_mean"]

                model.save(str(best_path))

                print(
                    "★ 最良モデル更新:",
                    str(best_path) + ".zip"
                )

        completed = True

        print("\n=== Ver.5 学習完了 ===")

        for row in rows:
            print(
                f"{row['actual_steps']:,} Step "
                f"| 旧={row['strict_mean']:.3f}秒 "
                f"| 新={row['relaxed_mean']:.3f}秒 "
                f"| 膝={row['knee_range_deg']:.1f}度 "
                f"| 3秒以上={row['over_3s']}/100"
            )

    except KeyboardInterrupt:
        print("\n手動中断しました")

    finally:
        if not completed:
            path = RESULTS / f"{prefix}_interrupted"
            model.save(str(path))
            print(
                "途中モデルを保存:",
                str(path) + ".zip"
            )

        env.close()


if __name__ == "__main__":
    main()
