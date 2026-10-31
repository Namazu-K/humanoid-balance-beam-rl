import argparse
import csv
from pathlib import Path
from collections import deque
from datetime import datetime
from unittest.mock import patch

import numpy as np
import stand_env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)

from train_balance_v4 import DynamicBalanceEnv


# =====================================
# 設定
# =====================================

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

FLAT_XML = Path(
    "humanoid_flat_feet_heel.xml"
).resolve()

TARGETS = [
    200_000,
    500_000,
    1_000_000,
    1_500_000,
    2_000_000,
]

BEST_PATH = RESULTS / "flat_scratch_best"
CSV_PATH = RESULTS / "flat_scratch_evaluation.csv"

# 既存のstand_env.pyは書き換えない
ORIGINAL_GYM_MAKE = stand_env.gym.make


def make_flat_env():
    """+5cm平足を使うVer.4環境"""

    def custom_make(*args, **kwargs):
        if args and args[0] == "Humanoid-v5":
            kwargs["xml_file"] = str(FLAT_XML)

        return ORIGINAL_GYM_MAKE(
            *args, **kwargs
        )

    with patch.object(
        stand_env.gym,
        "make",
        side_effect=custom_make
    ):
        env = DynamicBalanceEnv(
            episode_limit=20.0,
            push_v=0.0,
            push_w=0.0,
            tilt_deg=0.0,
        )

    return env


# =====================================
# 学習状況の表示
# =====================================

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

            stable = float(
                info.get(
                    "longest_stable_seconds",
                    0.0
                )
            )

            self.recent.append(stable)

            if self.episodes % 100 == 0:
                a = np.asarray(self.recent)

                print(
                    f"\n[平足学習] "
                    f"Step={self.num_timesteps:,}"
                    f" Episode={self.episodes}",
                    flush=True
                )
                print(
                    f"平均安定: {a.mean():.3f}秒"
                )
                print(
                    f"1秒以上: "
                    f"{np.mean(a >= 1)*100:.1f}%"
                )
                print(
                    f"2秒以上: "
                    f"{np.mean(a >= 2)*100:.1f}%"
                )
                print(
                    f"3秒以上: "
                    f"{np.mean(a >= 3)*100:.1f}%"
                )

        return True


# =====================================
# 評価
# =====================================

def evaluate(model, episodes=100):
    env = make_flat_env()

    stable_times = []
    survival_times = []

    try:
        for seed in range(
            1000, 1000 + episodes
        ):
            obs, _ = env.reset(seed=seed)

            while True:
                if model is None:
                    action = np.zeros(
                        env.action_space.shape,
                        dtype=np.float32
                    )
                else:
                    action, _ = model.predict(
                        obs,
                        deterministic=True
                    )

                obs, reward, done, trunc, info = (
                    env.step(action)
                )

                if done or trunc:
                    stable_times.append(
                        info["longest_stable_seconds"]
                    )
                    survival_times.append(
                        info["elapsed_seconds"]
                    )
                    break
    finally:
        env.close()

    a = np.asarray(stable_times)

    return {
        "mean_stable": float(a.mean()),
        "median_stable": float(np.median(a)),
        "max_stable": float(a.max()),
        "mean_survival": float(
            np.mean(survival_times)
        ),
        "over_1s": int(np.sum(a >= 1)),
        "over_2s": int(np.sum(a >= 2)),
        "over_3s": int(np.sum(a >= 3)),
        "over_5s": int(np.sum(a >= 5)),
    }


def show_result(label, r, episodes=100):
    print("\n" + "=" * 54)
    print(label)
    print("=" * 54)

    print(
        f"平均最長安定: "
        f"{r['mean_stable']:.3f}秒"
    )
    print(
        f"中央値: "
        f"{r['median_stable']:.3f}秒"
    )
    print(
        f"最高記録: "
        f"{r['max_stable']:.3f}秒"
    )
    print(
        f"平均生存: "
        f"{r['mean_survival']:.3f}秒"
    )
    print(
        f"1秒以上: {r['over_1s']}/{episodes}"
    )
    print(
        f"2秒以上: {r['over_2s']}/{episodes}"
    )
    print(
        f"3秒以上: {r['over_3s']}/{episodes}"
    )
    print(
        f"5秒以上: {r['over_5s']}/{episodes}"
    )


def save_csv(rows):
    with CSV_PATH.open(
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


# =====================================
# 動作確認
# =====================================

def smoke_test():
    env = make_flat_env()

    try:
        obs, _ = env.reset(seed=1000)

        print("観測:", obs.shape)
        print("行動:", env.action_space.shape)
        print(
            "初期胴体高さ:",
            f"{env.data.body('torso').xpos[2]:.3f}m"
        )

        for side in ("left", "right"):
            gid = env.model.geom(
                f"{side}_foot"
            ).id

            print(
                f"{side}_foot:",
                "位置=",
                np.round(
                    env.model.geom_pos[gid], 3
                ),
                "サイズ=",
                np.round(
                    env.model.geom_size[gid], 3
                )
            )

        assert obs.shape == (359,)
        assert env.action_space.shape == (17,)

        while True:
            action = np.zeros(
                env.action_space.shape,
                dtype=np.float32
            )

            obs, reward, done, trunc, info = (
                env.step(action)
            )

            if done or trunc:
                print(
                    "ゼロ操作の安定時間:",
                    f"{info['longest_stable_seconds']:.3f}秒"
                )
                print("終了理由:", info["reason"])
                break

        print("動作テスト成功！")

    finally:
        env.close()


# =====================================
# ゼロから学習
# =====================================

def train():
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    env = Monitor(
        make_flat_env(),
        filename=str(
            RESULTS
            / f"flat_scratch_{stamp}.monitor.csv"
        ),
        info_keywords=(
            "reason",
            "longest_stable_seconds",
        ),
    )

    # 学習済みモデルを読み込まない！
    # 重みをランダム初期化する。
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

    reporter = ProgressReport()

    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path=str(
            RESULTS / "flat_checkpoints"
        ),
        name_prefix="flat_scratch",
    )

    callbacks = CallbackList([
        reporter,
        checkpoint
    ])

    records = []
    best_mean = -float("inf")
    completed = False

    print("\n=== 平足 PPO ゼロから学習 ===")
    print("XML:", FLAT_XML)
    print("開始Step:", model.num_timesteps)
    print("目標: 200万Step")
    print("初期重み: ランダム")

    try:
        baseline = evaluate(None)
        show_result(
            "ゼロ操作・学習前",
            baseline
        )

        records.append({
            "target": 0,
            "actual_steps": 0,
            **baseline
        })
        save_csv(records)

        for target in TARGETS:
            remaining = (
                target - model.num_timesteps
            )

            print("\n" + "#" * 54)
            print(
                f"次の目標: {target:,} Step"
            )
            print(
                f"追加: {remaining:,} Step"
            )
            print("#" * 54)

            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    reset_num_timesteps=False,
                    callback=callbacks,
                )

            label = target // 1000

            path = (
                RESULTS
                / f"flat_scratch_{label}k"
            )

            # 評価前に必ず保存
            model.save(str(path))

            r = evaluate(model)

            show_result(
                f"累計 {model.num_timesteps:,} Step",
                r
            )

            records.append({
                "target": target,
                "actual_steps": model.num_timesteps,
                **r
            })

            save_csv(records)

            if r["mean_stable"] > best_mean:
                best_mean = r["mean_stable"]
                model.save(str(BEST_PATH))

                print(
                    "★ 最良モデル更新:",
                    str(BEST_PATH) + ".zip"
                )

        completed = True

        print("\n=== 全学習完了！ ===")

        for row in records:
            print(
                f"{row['target']//10000:3d}万"
                f" | 平均="
                f"{row['mean_stable']:.3f}秒"
                f" | 1秒以上="
                f"{row['over_1s']}/100"
                f" | 3秒以上="
                f"{row['over_3s']}/100"
            )

        print(
            "最良モデル:",
            str(BEST_PATH) + ".zip"
        )
        print("評価CSV:", CSV_PATH)

    except KeyboardInterrupt:
        print("\n手動中断しました")

    finally:
        if not completed:
            path = RESULTS / "flat_scratch_interrupted"
            model.save(str(path))

            print(
                "途中モデル保存:",
                str(path) + ".zip"
            )

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true"
    )

    args = parser.parse_args()

    if not FLAT_XML.exists():
        raise FileNotFoundError(
            f"平足XMLがありません: {FLAT_XML}"
        )

    if args.test:
        smoke_test()
    else:
        train()
