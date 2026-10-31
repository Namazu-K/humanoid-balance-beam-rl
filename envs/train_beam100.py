import sys
sys.modules["triton"] = None

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
import mujoco

import stand_env
from train_flat_v5_sparse import FlatBalanceV5
from train_flat_scratch import ORIGINAL_GYM_MAKE
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


ROOT = Path("results/beam100")
ROOT.mkdir(parents=True, exist_ok=True)

SOURCE_XML = Path("humanoid_flat_feet_heel.xml")
SOURCE_MODEL = Path("results/v5_to_10m/model_5500k.zip")

BEAM_LENGTH = 100.0
BEAM_HEIGHT = 1.5

# 段階: 目標距離 → (台幅, 制限時間)
STAGES = {
    1:   (1.0, 45.0),
    2:   (1.0, 60.0),
    5:   (0.8, 90.0),
    10:  (0.8, 120.0),
    25:  (0.6, 180.0),
    50:  (0.6, 240.0),
    100: (0.4, 300.0),
}

ORDER = list(STAGES)


# ==========================================
# 平均台XML生成
# ==========================================

def create_beam_xml(width):
    if not SOURCE_XML.exists():
        raise FileNotFoundError(SOURCE_XML)

    filename = ROOT / f"beam_width_{width:.2f}.xml"

    tree = ET.parse(SOURCE_XML)
    root = tree.getroot()

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("worldbody がありません")

    # 既存のfloorは1.5m下の地面にする
    floor = world.find("./geom[@name='floor']")
    if floor is None:
        raise RuntimeError("元XMLのfloorが見つかりません")

    floor.set("name", "ground_below")
    floor.set("pos", f"0 0 {-BEAM_HEIGHT}")

    # 平均台自体をfloorと名付ける。
    # 既存の足接地判定をそのまま使うため。
    # X方向 -1m ～ 101m、長さ102mの支持面。
    # ゴールは開始位置から100m。
    beam = ET.Element("geom", {
        "name": "floor",
        "type": "box",
        "pos": "50 0 -0.125",
        "size": f"51 {width / 2:.5f} 0.125",
        "friction": "1.5 0.1 0.1",
        "condim": "3",
        "rgba": "0.75 0.53 0.27 1",
    })

    world.insert(0, beam)
    tree.write(filename, encoding="unicode")

    return filename.resolve()


# ==========================================
# 平均台上での歩行タスク
# ==========================================

class BeamWalkingEnv(gym.Wrapper):

    def __init__(self, base, goal_m, width):
        super().__init__(base)

        self.base = base
        self.goal_m = float(goal_m)
        self.width = float(width)

        self.start_x = 0.0
        self.previous_x = 0.0
        self.maximum_x = 0.0

        # 平均台の状態を観測に7次元追加
        original = base.observation_space

        self.observation_space = gym.spaces.Box(
            low=np.concatenate([
                original.low,
                np.full(7, -np.inf)
            ]).astype(np.float32),
            high=np.concatenate([
                original.high,
                np.full(7, np.inf)
            ]).astype(np.float32),
            dtype=np.float32,
        )

    def position(self):
        return float(self.base.data.qpos[0])

    def extra_observation(self):
        d = self.base.data

        x = self.position()
        y = float(d.qpos[1])
        vx = float(d.qvel[0])
        vy = float(d.qvel[1])

        travelled = x - self.start_x
        remaining = self.goal_m - travelled

        return np.asarray([
            np.clip(y / (self.width / 2), -5, 5),
            np.clip(vx / 2.0, -5, 5),
            np.clip(vy / 2.0, -5, 5),
            travelled / BEAM_LENGTH,
            remaining / max(self.goal_m, 1.0),
            self.width,
            self.goal_m / BEAM_LENGTH,
        ], dtype=np.float32)

    def augment(self, obs):
        return np.concatenate([
            np.asarray(obs, dtype=np.float32),
            self.extra_observation(),
        ])

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(
            seed=seed, options=options
        )

        self.start_x = self.position()
        self.previous_x = self.start_x
        self.maximum_x = self.start_x

        return self.augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.base.step(action)
        )

        x = self.position()
        dx = x - self.previous_x

        self.previous_x = x
        self.maximum_x = max(self.maximum_x, x)

        stable = bool(
            info.get("relaxed_stable", False)
        )

        # ==================================
        # 歩行報酬
        # ==================================
        # 動けなければ時間コストを負担。
        # 足場上で安定して前進すると報酬。
        # 後退は軽く減点。
        reward -= 2.0 * self.base.dt

        if stable:
            reward += 40.0 * max(0.0, dx)
            reward -= 10.0 * max(0.0, -dx)
        else:
            reward -= 2.0 * abs(dx)

        distance = max(
            0.0,
            self.maximum_x - self.start_x,
        )

        # 目標位置で安全に立っていること
        # を成功条件とする
        success = bool(
            x - self.start_x >= self.goal_m
            and stable
        )

        if success:
            reward += 100.0
            terminated = True
            info["reason"] = "goal"

        info["distance_m"] = float(distance)
        info["current_x_m"] = float(
            x - self.start_x
        )
        info["is_success"] = success
        info["beam_width_m"] = self.width

        return (
            self.augment(obs),
            float(reward),
            terminated,
            truncated,
            info,
        )


def make_beam_env(stage):
    width, seconds = STAGES[stage]
    xml_path = create_beam_xml(width)

    def custom_make(*args, **kwargs):
        if args and args[0] == "Humanoid-v5":
            kwargs["xml_file"] = str(xml_path)

        return ORIGINAL_GYM_MAKE(
            *args, **kwargs
        )

    with patch.object(
        stand_env.gym,
        "make",
        side_effect=custom_make,
    ):
        base = FlatBalanceV5()

    # 元の20秒制限を解除
    base.episode_limit = seconds

    return BeamWalkingEnv(
        base,
        goal_m=stage,
        width=width,
    )


# ==========================================
# 旧モデルの知識を新モデルへ引き継ぐ
# ==========================================

def transfer_standing_policy(env):
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(SOURCE_MODEL)

    old = PPO.load(
        str(SOURCE_MODEL), device="cpu"
    )

    # 新しい7次元の観測に対応したPPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.15,
        target_kl=0.02,
        ent_coef=0.0,
        policy_kwargs=old.policy_kwargs,
        device="cpu",
        verbose=1,
        seed=42,
    )

    old_weights = old.policy.state_dict()
    new_weights = model.policy.state_dict()

    expanded_layers = []

    for key, src in old_weights.items():
        if key not in new_weights:
            raise RuntimeError(
                f"転送先に層がない: {key}"
            )

        dst = new_weights[key]

        if src.shape == dst.shape:
            dst.copy_(src)

        elif (
            src.ndim == 2
            and dst.ndim == 2
            and src.shape[0] == dst.shape[0]
            and dst.shape[1] == src.shape[1] + 7
        ):
            # 旧359次元分の重みを維持。
            # 新しい7次元分はゼロ初期化。
            dst.zero_()
            dst[:, :src.shape[1]].copy_(src)
            expanded_layers.append(key)

        else:
            raise RuntimeError(
                f"重み形状が不一致: {key} "
                f"{tuple(src.shape)} -> "
                f"{tuple(dst.shape)}"
            )

    if len(expanded_layers) != 2:
        raise RuntimeError(
            "追加観測に対応する2層の拡張に失敗: "
            + str(expanded_layers)
        )

    model.policy.load_state_dict(new_weights)

    # 新タスクのオプティマイザーは新規作成。
    # 旧モデルの政策だけを転送する。
    model.num_timesteps = old.num_timesteps

    print("\n=== 既存モデルの引き継ぎ完了 ===")
    print("旧観測:", old.observation_space.shape)
    print("新観測:", model.observation_space.shape)
    print("拡張層:", expanded_layers)
    print(
        "旧学習累計:",
        f"{old.num_timesteps:,}"
    )

    return model


def model_path(stage):
    return ROOT / f"stage_{stage}m.zip"


def load_model_for_stage(stage, env):
    own = model_path(stage)

    if own.exists():
        print("この段階の続きから学習:", own)
        return PPO.load(
            str(own), env=env, device="cpu"
        )

    index = ORDER.index(stage)

    if index == 0:
        return transfer_standing_policy(env)

    previous_stage = ORDER[index - 1]
    previous = model_path(previous_stage)

    if not previous.exists():
        raise FileNotFoundError(
            f"先に{previous_stage}m段階の"
            f"モデルを作成してください: {previous}"
        )

    print("前段階から引き継ぎ:", previous)

    return PPO.load(
        str(previous),
        env=env,
        device="cpu",
        learning_rate=3e-5,
    )


# ==========================================
# モデル評価
# ==========================================

def evaluate(model, stage, episodes):
    env = make_beam_env(stage)
    rows = []

    try:
        for i in range(episodes):
            seed = 3000 + i
            obs, _ = env.reset(seed=seed)

            while True:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, done, trunc, info = (
                    env.step(action)
                )

                if done or trunc:
                    break

            rows.append({
                "seed": seed,
                "distance_m":
                    float(info["distance_m"]),
                "success":
                    int(info["is_success"]),
                "survival_s":
                    float(info["elapsed_seconds"]),
                "strict_stable_s": float(
                    info["longest_stable_seconds"]
                ),
                "relaxed_stable_s": float(
                    info["longest_relaxed_seconds"]
                ),
                "reason":
                    str(info.get("reason", "")),
            })

    finally:
        env.close()

    distances = np.asarray([
        r["distance_m"] for r in rows
    ])

    successes = sum(
        r["success"] for r in rows
    )

    summary = {
        "stage_m": stage,
        "width_m": STAGES[stage][0],
        "episodes": episodes,
        "successes": successes,
        "success_rate_pct":
            100.0 * successes / episodes,
        "mean_distance_m":
            float(np.mean(distances)),
        "median_distance_m":
            float(np.median(distances)),
        "max_distance_m":
            float(np.max(distances)),
        "mean_survival_s": float(
            np.mean([r["survival_s"] for r in rows])
        ),
        "mean_relaxed_stable_s": float(
            np.mean([
                r["relaxed_stable_s"]
                for r in rows
            ])
        ),
    }

    print("\n" + "=" * 55)
    print(f"【{stage}m歩行評価】")
    print("=" * 55)

    print(
        f"成功率: {successes}/{episodes} "
        f"({summary['success_rate_pct']:.1f}%)"
    )
    print(
        f"平均前進距離: "
        f"{summary['mean_distance_m']:.3f}m"
    )
    print(
        f"最大前進距離: "
        f"{summary['max_distance_m']:.3f}m"
    )
    print(
        f"平均生存: "
        f"{summary['mean_survival_s']:.2f}秒"
    )

    detail_path = (
        ROOT / f"stage_{stage}m_episodes.csv"
    )

    with detail_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = ROOT / "progress.csv"

    existing = []
    if summary_path.exists():
        with summary_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as f:
            existing = list(csv.DictReader(f))

    existing.append({
        "steps": model.num_timesteps,
        **summary,
    })

    with summary_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(existing[0].keys())
        )
        writer.writeheader()
        writer.writerows(existing)

    return summary


# ==========================================
# 動画録画
# ==========================================

def record_video(model, stage, seed=5000):
    import imageio.v2 as imageio

    env = make_beam_env(stage)
    renderer = None
    writer = None

    path = ROOT / (
        f"stage_{stage}m_seed{seed}.mp4"
    )

    try:
        obs, _ = env.reset(seed=seed)

        renderer = mujoco.Renderer(
            env.base.model,
            width=640,
            height=480,
        )

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 135
        camera.elevation = -15
        camera.distance = 3.8

        writer = imageio.get_writer(
            str(path),
            fps=30,
            codec="libx264",
            quality=8,
        )

        def capture():
            torso = env.base.data.body("torso").xpos

            # 移動するロボットを追跡
            camera.lookat[:] = torso + np.array([
                0.0, 0.0, -0.4
            ])

            renderer.update_scene(
                env.base.data,
                camera=camera,
            )

            writer.append_data(
                np.array(
                    renderer.render(),
                    copy=True,
                )
            )

        capture()
        next_frame = 1.0 / 30.0

        while True:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, done, trunc, info = (
                env.step(action)
            )

            if env.base.elapsed >= next_frame:
                capture()
                next_frame += 1.0 / 30.0

            if done or trunc:
                break

        print("\n動画保存:", path)
        print(
            "最大前進距離:",
            f"{info['distance_m']:.3f}m"
        )
        print(
            "生存時間:",
            f"{info['elapsed_seconds']:.2f}秒"
        )

    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()


# ==========================================
# 実行
# ==========================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage", type=int, default=1,
        choices=ORDER
    )
    parser.add_argument(
        "--steps", type=int, default=500_000
    )
    parser.add_argument(
        "--episodes", type=int, default=20
    )
    parser.add_argument(
        "--test", action="store_true"
    )
    parser.add_argument(
        "--eval-only", action="store_true"
    )
    parser.add_argument(
        "--video", action="store_true"
    )

    args = parser.parse_args()

    env = make_beam_env(args.stage)

    try:
        obs, _ = env.reset(seed=42)

        print("\n=== 平均台設定 ===")
        print("平均台の長さ:", BEAM_LENGTH, "m")
        print("平均台の高さ:", BEAM_HEIGHT, "m")
        print("平均台の幅:", env.width, "m")
        print("今回の目標:", env.goal_m, "m")
        print("制限時間:", env.base.episode_limit, "秒")
        print("観測次元:", obs.shape)
        print("行動次元:", env.action_space.shape)

        assert obs.shape == (366,)
        assert env.action_space.shape == (17,)

        if args.test:
            for _ in range(15):
                a = np.zeros(
                    env.action_space.shape,
                    dtype=np.float32
                )

                obs, r, done, trunc, info = (
                    env.step(a)
                )

                if done or trunc:
                    break

            print("平均台環境テスト成功")
            return

        if args.eval_only:
            path = model_path(args.stage)

            if not path.exists():
                raise FileNotFoundError(path)

            model = PPO.load(
                str(path), env=env, device="cpu"
            )

        else:
            model = load_model_for_stage(
                args.stage, env
            )

            monitored = Monitor(
                env,
                filename=str(
                    ROOT / f"stage_{args.stage}m_monitor.csv"
                ),
                info_keywords=(
                    "distance_m", "is_success"
                ),
            )

            model.set_env(monitored)

            checkpoints = CheckpointCallback(
                save_freq=100_000,
                save_path=str(
                    ROOT / "checkpoints"
                ),
                name_prefix=(
                    f"beam_{args.stage}m"
                ),
            )

            print("\n=== 歩行学習開始 ===")

            try:
                model.learn(
                    total_timesteps=args.steps,
                    reset_num_timesteps=False,
                    callback=checkpoints,
                )

            except KeyboardInterrupt:
                emergency = ROOT / (
                    f"stage_{args.stage}m_interrupted.zip"
                )
                model.save(str(emergency))
                print("途中保存:", emergency)
                return

            finally:
                monitored.close()

            output = model_path(args.stage)
            model.save(str(output))

            print("\nモデル保存:", output)

        evaluate(
            model, args.stage, args.episodes
        )

        if args.video:
            record_video(
                model, args.stage, seed=5000
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()
