import csv
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from stand_env import StandEnv


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

START_MODEL = RESULTS / "phase0_hold1s_800k.zip"
BEST_MODEL = RESULTS / "phase0_endurance_best"
CSV_PATH = RESULTS / "phase0_endurance_evaluation.csv"

TARGETS = [
    1_000_000,
    1_200_000,
    1_500_000,
    2_000_000,
]


class StandEnduranceEnv(StandEnv):
    """
    Phase 0 Ver.3:
    1秒で終了せず、倒れるまで直立を続ける。
    """

    def __init__(self, episode_limit=20.0):
        super().__init__(hold_seconds=1.0)

        self.episode_limit = float(episode_limit)

        # 重要：
        # 親クラスのmax_seconds=6.0は変更しない。
        # 旧モデルが学習した時間観測の尺度を維持する。
        # 実際の制限時間はepisode_limitで管理する。

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

        # 親クラスの1秒成功判定を使わず、
        # 基礎MuJoCo環境を直接1ステップ進める。
        raw_obs, _, base_done, base_truncated, _ = (
            self.env.step(action)
        )

        self.elapsed += self.dt

        z, angle, posture, height, contact = (
            self._status()
        )

        if contact:
            self.air_seconds = 0.0
        else:
            self.air_seconds += self.dt

        # これまでと同じ安定姿勢の定義
        stable = (
            angle <= 30.0
            and z >= self.initial_z - 0.20
            and contact
        )

        if stable:
            self.stable_seconds += self.dt
            self.longest_stable_seconds = max(
                self.longest_stable_seconds,
                self.stable_seconds
            )
        else:
            self.stable_seconds = 0.0

        # 姿勢を崩したときの終了条件
        fallen = (
            z < self.initial_z - 0.55
            or angle > 80.0
            or self.air_seconds >= 0.8
        )

        angular_speed = float(
            np.linalg.norm(self.data.qvel[3:6])
        )

        effort = float(np.mean(
            (action / 0.4) ** 2
        ))

        xy = np.asarray(
            self.data.body("torso").xpos[:2]
        )
        drift = float(np.linalg.norm(
            xy - self.initial_xy
        ))

        # 毎ステップの持続報酬。
        # 1秒達成ボーナスは設けない。
        reward = self.dt * (
            4.0 * posture * height
            + 2.0 * float(stable)
            + 0.5 * float(contact)
            - 0.06 * effort
            - 0.08 * angular_speed
            - 0.5 * drift
        )

        if fallen:
            reward -= 5.0

        terminated = bool(base_done or fallen)

        # 時間上限は転倒ではなくtruncated
        timed_out = self.elapsed >= self.episode_limit
        truncated = bool(
            base_truncated
            or (timed_out and not terminated)
        )

        if fallen:
            reason = "fall"
        elif base_done:
            reason = "environment"
        elif truncated:
            reason = "time_limit"
        else:
            reason = "running"

        obs = self._observation(
            raw_obs, posture, contact
        )

        info = {
            "phase": 0,
            "reason": reason,
            "elapsed_seconds": float(self.elapsed),
            "stable_seconds": float(self.stable_seconds),
            "longest_stable_seconds": float(
                self.longest_stable_seconds
            ),
            "torso_height_m": float(z),
            "torso_angle_deg": float(angle),
            "posture_quality": float(posture),
            "height_quality": float(height),
            "feet_contact": bool(contact),
            "reached_1s": bool(
                self.longest_stable_seconds >= 1.0
            ),
            "reached_3s": bool(
                self.longest_stable_seconds >= 3.0
            ),
            "reached_5s": bool(
                self.longest_stable_seconds >= 5.0
            ),
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )


class TrainingReport(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = 0
        self.recent_stable = deque(maxlen=100)
        self.recent_1s = deque(maxlen=100)
        self.recent_3s = deque(maxlen=100)

    def _on_step(self):
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episodes += 1

            stable = float(
                info.get("longest_stable_seconds", 0.0)
            )

            self.recent_stable.append(stable)
            self.recent_1s.append(
                int(stable >= 1.0)
            )
            self.recent_3s.append(
                int(stable >= 3.0)
            )

            if self.episodes % 100 == 0:
                print(
                    f"\n累計Step={self.model.num_timesteps:,}"
                    f" | 追加Episode={self.episodes}"
                )
                print(
                    "直近100回の平均安定:",
                    f"{np.mean(self.recent_stable):.3f}秒"
                )
                print(
                    "1秒以上:",
                    f"{np.mean(self.recent_1s)*100:.1f}%"
                )
                print(
                    "3秒以上:",
                    f"{np.mean(self.recent_3s)*100:.1f}%"
                )

        return True


def evaluate(model, episodes=100, seed_start=1000):
    env = StandEnduranceEnv(episode_limit=20.0)

    stable_times = []
    survival_times = []
    rewards = []
    timeouts = 0

    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0

            while True:
                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                obs, reward, done, truncated, info = (
                    env.step(action)
                )
                total_reward += reward

                if done or truncated:
                    stable_times.append(
                        info["longest_stable_seconds"]
                    )
                    survival_times.append(
                        info["elapsed_seconds"]
                    )
                    rewards.append(total_reward)
                    timeouts += int(
                        info["reason"] == "time_limit"
                    )
                    break
    finally:
        env.close()

    a = np.asarray(stable_times)

    return {
        "mean_stable": float(a.mean()),
        "max_stable": float(a.max()),
        "mean_survival": float(
            np.mean(survival_times)
        ),
        "mean_reward": float(np.mean(rewards)),
        "over_1s": int(np.sum(a >= 1.0)),
        "over_2s": int(np.sum(a >= 2.0)),
        "over_3s": int(np.sum(a >= 3.0)),
        "over_5s": int(np.sum(a >= 5.0)),
        "over_10s": int(np.sum(a >= 10.0)),
        "time_limit_count": timeouts,
    }


def print_score(label, s):
    print("\n" + "=" * 55)
    print(label)
    print("=" * 55)
    print(f"平均最長安定: {s['mean_stable']:.3f}秒")
    print(f"最高安定時間: {s['max_stable']:.3f}秒")
    print(f"平均生存時間: {s['mean_survival']:.3f}秒")
    print(f"平均報酬: {s['mean_reward']:.3f}")
    print(f"1秒以上: {s['over_1s']}/100")
    print(f"2秒以上: {s['over_2s']}/100")
    print(f"3秒以上: {s['over_3s']}/100")
    print(f"5秒以上: {s['over_5s']}/100")
    print(f"10秒以上: {s['over_10s']}/100")
    print(
        f"制限時間到達: {s['time_limit_count']}/100"
    )


def save_csv(records):
    with CSV_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    if not START_MODEL.exists():
        raise FileNotFoundError(
            f"開始モデルがありません: {START_MODEL}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = Monitor(
        StandEnduranceEnv(episode_limit=20.0),
        filename=str(
            RESULTS / f"endurance_{stamp}.monitor.csv"
        ),
        info_keywords=(
            "reason",
            "longest_stable_seconds",
            "reached_1s",
            "reached_3s",
            "reached_5s",
        ),
    )

    # 80万モデルの重みと探索設定を引き継ぐ
    model = PPO.load(
        str(START_MODEL),
        env=env,
        device="cpu",
        gamma=0.999,
        target_kl=0.02,
    )

    report = TrainingReport()
    records = []
    completed = False

    print("\n=== Phase 0 持続直立学習 ===")
    print("開始モデル:", START_MODEL)
    print("現在Step:", model.num_timesteps)
    print("探索標準偏差は保存モデルを継承")
    print("gamma:", model.gamma)
    print("最大試行時間: 20秒")

    try:
        # 学習開始前の基準値
        score = evaluate(model)
        print_score("開始時点：80万モデル", score)

        records.append({
            "target": 800_000,
            "actual_steps": model.num_timesteps,
            **score,
        })

        best_mean = score["mean_stable"]
        model.save(str(BEST_MODEL))
        save_csv(records)

        for target in TARGETS:
            remaining = target - model.num_timesteps

            if remaining > 0:
                print(
                    f"\n次の目標：{target:,}ステップ"
                )

                model.learn(
                    total_timesteps=remaining,
                    reset_num_timesteps=False,
                    callback=report,
                )

            label = target // 1000
            path = (
                RESULTS / f"phase0_endurance_{label}k"
            )

            # 評価前に必ずスナップショット保存
            model.save(str(path))

            score = evaluate(model)

            print_score(
                f"累計 {model.num_timesteps:,} Step",
                score,
            )

            records.append({
                "target": target,
                "actual_steps": model.num_timesteps,
                **score,
            })
            save_csv(records)

            # 1秒到達回数ではなく平均持続時間を優先
            if score["mean_stable"] > best_mean:
                best_mean = score["mean_stable"]
                model.save(str(BEST_MODEL))
                print(
                    "★ 最良モデルを更新:",
                    str(BEST_MODEL) + ".zip"
                )

        completed = True

        print("\n=== 学習完了 ===")
        for row in records:
            print(
                f"{row['target']//10000:3d}万Step "
                f"| 平均={row['mean_stable']:.3f}秒 "
                f"| 3秒以上={row['over_3s']}/100 "
                f"| 5秒以上={row['over_5s']}/100"
            )

        print(
            "最良モデル:",
            str(BEST_MODEL) + ".zip"
        )
        print("評価CSV:", CSV_PATH)

    except KeyboardInterrupt:
        print("\n学習を中断しました。")

    finally:
        if not completed:
            path = RESULTS / "phase0_endurance_interrupted"
            model.save(str(path))
            print(
                "中断時モデル:",
                str(path) + ".zip"
            )

        env.close()


if __name__ == "__main__":
    main()
