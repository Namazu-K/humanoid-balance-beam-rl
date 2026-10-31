import sys
sys.modules['triton'] = None

import argparse
import csv
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import stand_env
import train_beam100 as previous
from train_flat_v5_sparse import FlatBalanceV5
from train_flat_scratch import ORIGINAL_GYM_MAKE

OUT = Path('results/beam110')
OUT.mkdir(parents=True, exist_ok=True)
XML_SOURCE = Path('humanoid_flat_feet_heel.xml')
STANDING_MODEL = Path('results/v5_to_10m/model_5500k.zip')
OLD_BEAM_MODEL = Path('results/beam100_fallok/stage_1m.zip')
BEAM_LENGTH = 110.0
GOAL_X = 100.0
CHECKPOINTS = list(range(5, 101, 5))
REACH_MARKS = [1] + CHECKPOINTS


def beam_xml(width):
    tree = ET.parse(XML_SOURCE)
    root = tree.getroot()
    world = root.find('worldbody')
    floor = world.find("./geom[@name='floor']")
    if floor is None:
        raise RuntimeError('元XMLのfloorが見つかりません')
    floor.set('name', 'ground_below')
    floor.set('pos', '0 0 -1.5')
    ET.SubElement(world, 'geom', {
        'name': 'floor', 'type': 'box',
        'pos': '50 0 -0.125',
        'size': f'55 {width / 2:.5f} 0.125',
        'friction': '1.5 0.1 0.1', 'condim': '3',
        'rgba': '0.72 0.52 0.30 1',
    })
    path = OUT / f'beam110_width_{width:.2f}.xml'
    tree.write(path, encoding='unicode')
    return path.resolve()


class Beam110Env(previous.BeamWalkingEnv):
    """観測は旧平均台と互換の366次元。終了・報酬は全面改訂。"""

    def __init__(self, base, width, seconds, stall_seconds):
        super().__init__(base, goal_m=GOAL_X, width=width)
        self.seconds = seconds
        self.stall_seconds = stall_seconds
        m = base.model
        self.beam_id = int(m.geom('floor').id)
        self.ground_id = int(m.geom('ground_below').id)
        self.robot_geoms = np.where(np.asarray(m.geom_bodyid) != 0)[0]
        self.torso_id = int(m.body('torso').id)
        base.episode_limit = seconds
        native = base.env.unwrapped
        if hasattr(native, '_terminate_when_unhealthy'):
            native._terminate_when_unhealthy = False

    def front_x(self):
        """全ロボットgeomの実際の形状を考慮した先端X座標。"""
        m, d = self.base.model, self.base.data
        best = -float('inf')
        for gid in self.robot_geoms:
            gid = int(gid)
            t = int(m.geom_type[gid])
            r = d.geom_xmat[gid].reshape(3, 3)[0]
            size = m.geom_size[gid]
            if t == int(mujoco.mjtGeom.mjGEOM_BOX):
                extension = float(np.sum(np.abs(r) * size[:3]))
            elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                extension = float(size[0])
            elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                extension = float(size[0] + size[1] * abs(r[2]))
            elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                extension = float(
                    size[0] * np.sqrt(max(0.0, 1.0-r[2]**2))
                    + size[1] * abs(r[2])
                )
            elif t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                extension = float(np.sqrt(np.sum((r * size[:3]) ** 2)))
            else:
                extension = float(m.geom_rbound[gid])
            best = max(best, float(d.geom_xpos[gid, 0]) + extension)
        return best

    def contacts(self):
        m, d = self.base.model, self.base.data
        beam_touch, ground_touch, hand_touch = False, False, False
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            for surface, name in ((self.beam_id, 'beam'),
                                  (self.ground_id, 'ground')):
                if a == surface or b == surface:
                    other = b if a == surface else a
                    if int(m.geom_bodyid[other]) == 0:
                        continue
                    if name == 'beam':
                        beam_touch = True
                        geom_name = mujoco.mj_id2name(
                            m, mujoco.mjtObj.mjOBJ_GEOM, other
                        ) or ''
                        hand_touch |= 'hand' in geom_name.lower()
                    else:
                        ground_touch = True
        return beam_touch, ground_touch, hand_touch

    def extra_observation(self):
        d = self.base.data
        x = self.position()
        travelled = x - self.start_x
        return np.asarray([
            np.clip(float(d.qpos[1]) / (self.width / 2), -5, 5),
            np.clip(float(d.qvel[0]) / 2, -5, 5),
            np.clip(float(d.qvel[1]) / 2, -5, 5),
            travelled / BEAM_LENGTH,
            (GOAL_X - travelled) / GOAL_X,
            self.width,
            GOAL_X / BEAM_LENGTH,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.best_front = max(0.0, self.front_x())
        self.next_cp_index = 1
        self.no_contact_seconds = 0.0
        self.stall_clock = 0.0
        self.stall_reference = self.best_front
        self.best_com_x = float(self.base.data.subtree_com[self.torso_id, 0])
        self.best_rear_foot_x = min(
            float(self.base.data.geom('left_foot').xpos[0]),
            float(self.base.data.geom('right_foot').xpos[0]),
        )
        self.finished = False
        return obs, info

    def step(self, action):
        # 旧環境の「転倒した」フラグと報酬は採用しない
        obs, _, _, _, info = self.base.step(action)
        info = dict(info)
        dt = float(self.base.dt)
        beam_touch, ground_touch, hand_touch = self.contacts()
        self.no_contact_seconds = (
            0.0 if beam_touch else self.no_contact_seconds + dt
        )
        com = self.base.data.subtree_com[self.torso_id]
        com_z = float(com[2])
        off_beam = (
            ground_touch
            or self.no_contact_seconds >= 1.2
            or (com_z < -0.4 and self.no_contact_seconds > 0.2)
        )

        # 落ちていない間だけ記録を更新。前後振動の重複加点なし。
        progress_pts = 0.0
        checkpoint_pts = 0.0
        if not off_beam:
            current_front = min(GOAL_X, self.front_x())
            new_best = max(self.best_front, current_front)
            progress_pts = 2.0 * (new_best - self.best_front)
            self.best_front = new_best
            self.best_com_x = max(self.best_com_x, float(com[0]))
            rear_foot = min(
                float(self.base.data.geom('left_foot').xpos[0]),
                float(self.base.data.geom('right_foot').xpos[0]),
            )
            self.best_rear_foot_x = max(self.best_rear_foot_x, rear_foot)
            while (self.next_cp_index <= 20
                   and self.best_front >= self.next_cp_index * 5):
                cp_metres = self.next_cp_index * 5
                checkpoint_pts += (cp_metres / 10.0) * 5.0
                self.next_cp_index += 1

        self.stall_clock += dt
        if self.best_front >= self.stall_reference + 0.10:
            self.stall_reference = self.best_front
            self.stall_clock = 0.0

        # 姿勢には一切点を与えない
        reward = progress_pts + checkpoint_pts - 0.02 * dt
        terminated = False
        truncated = False
        reason = 'continuing'
        success = False

        if off_beam:
            reward -= 12.0
            terminated = True
            reason = 'off_beam'
        elif self.best_front >= GOAL_X:
            reward += 300.0
            terminated = True
            reason = 'goal'
            success = True
        elif self.stall_clock >= self.stall_seconds:
            reward -= 10.0
            terminated = True
            reason = 'stalled'
        elif float(self.base.elapsed) >= self.seconds:
            truncated = True
            reason = 'time_limit'

        info.update({
            'reason': reason,
            'is_success': success,
            'distance_m': float(self.best_front),
            'front_x_m': float(self.front_x()),
            'com_max_x_m': float(self.best_com_x),
            'rear_foot_max_x_m': float(self.best_rear_foot_x),
            'checkpoints_passed': self.next_cp_index - 1,
            'progress_points': float(progress_pts),
            'checkpoint_points': float(checkpoint_pts),
            'beam_contact': bool(beam_touch),
            'hand_contact': bool(hand_touch),
            'stall_seconds': float(self.stall_clock),
        })
        return self.augment(obs), float(reward), terminated, truncated, info


def make_env(width, seconds, stall_seconds):
    xml = beam_xml(width)

    def patched_make(*args, **kwargs):
        if args and args[0] == 'Humanoid-v5':
            kwargs['xml_file'] = str(xml)
        return ORIGINAL_GYM_MAKE(*args, **kwargs)

    with patch.object(stand_env.gym, 'make', side_effect=patched_make):
        base = FlatBalanceV5()
    return Beam110Env(base, width=width, seconds=seconds,
                      stall_seconds=stall_seconds)


def append_csv(path, rows):
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8-sig' if not exists else 'utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def evaluate(model, args, label):
    env = make_env(args.width, args.seconds, args.stall_seconds)
    records = []
    try:
        for i in range(args.episodes):
            obs, _ = env.reset(seed=3000+i)
            hand_frames, frames = 0, 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(action)
                frames += 1
                hand_frames += int(info['hand_contact'])
                if done or trunc:
                    break
            records.append({
                'label': label,
                'steps': model.num_timesteps,
                'seed': 3000+i,
                'max_front_m': info['distance_m'],
                'max_com_m': info['com_max_x_m'],
                'max_rear_foot_m': info['rear_foot_max_x_m'],
                'survival_s': float(info['elapsed_seconds']),
                'end_reason': info['reason'],
                'hand_contact_pct': 100*hand_frames/max(1,frames),
                **{f'reach_{d}m': int(info['distance_m'] >= d)
                   for d in REACH_MARKS},
            })
    finally:
        env.close()

    summary = {
        'label': label,
        'steps': model.num_timesteps,
        'episodes': len(records),
        'width_m': args.width,
        'mean_front_m': float(np.mean([r['max_front_m'] for r in records])),
        'max_front_m': float(max(r['max_front_m'] for r in records)),
        'mean_com_m': float(np.mean([r['max_com_m'] for r in records])),
        'mean_rear_foot_m': float(np.mean([r['max_rear_foot_m'] for r in records])),
        'mean_survival_s': float(np.mean([r['survival_s'] for r in records])),
        'mean_hand_contact_pct': float(np.mean([r['hand_contact_pct'] for r in records])),
        'fall_count': sum(r['end_reason'] == 'off_beam' for r in records),
        'stall_count': sum(r['end_reason'] == 'stalled' for r in records),
        **{f'reach_{d}m_pct': 100 * np.mean([r[f'reach_{d}m'] for r in records])
           for d in REACH_MARKS},
    }
    append_csv(OUT / 'evaluation_episodes.csv', records)
    append_csv(OUT / 'evaluation_summary.csv', [summary])
    print(f"\n[{label}] Step={model.num_timesteps:,}, "
          f"平均前進={summary['mean_front_m']:.3f}m, "
          f"最大={summary['max_front_m']:.3f}m")
    print(f"  1m={summary['reach_1m_pct']:.0f}%  "
          f"5m={summary['reach_5m_pct']:.0f}%  "
          f"10m={summary['reach_10m_pct']:.0f}%  "
          f"100m={summary['reach_100m_pct']:.0f}%")
    print(f"  重心の最大X={summary['mean_com_m']:.3f}m、"
          f"後方の足の最大X={summary['mean_rear_foot_m']:.3f}m、"
          f"落下={summary['fall_count']}/{len(records)}")
    return summary


def record_video(model, args, label, seed=5000):
    env = make_env(args.width, args.seconds, args.stall_seconds)
    renderer, writer = None, None
    file = OUT / f'{label}_seed{seed}.mp4'
    try:
        obs, _ = env.reset(seed=seed)
        renderer = mujoco.Renderer(env.base.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 4.0
        camera.azimuth = 135
        camera.elevation = -15
        writer = imageio.get_writer(str(file), fps=20, codec='libx264', quality=8)

        def frame():
            camera.lookat[:] = (env.base.data.body('torso').xpos
                                + np.array([0.0, 0.0, -0.4]))
            renderer.update_scene(env.base.data, camera=camera)
            writer.append_data(np.array(renderer.render(), copy=True))

        frame()
        next_frame = 1.0 / 20
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(action)
            if float(env.base.elapsed) >= next_frame:
                frame()
                next_frame += 1.0 / 20
            if done or trunc:
                break
        print(f"録画: {file} ／最前部 {info['distance_m']:.3f}m、終了 {info['reason']}")
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=200_000,
                   help='今回の追加学習ステップ数')
    p.add_argument('--eval-every', type=int, default=100_000)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--width', type=float, default=1.0)
    p.add_argument('--seconds', type=float, default=300.0)
    p.add_argument('--stall-seconds', type=float, default=30.0)
    p.add_argument('--init', choices=['standing', 'beam1m'], default='standing')
    p.add_argument('--test', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    if args.steps < 0 or args.eval_every < 1 or args.episodes < 1:
        p.error('steps>=0、eval-every>=1、episodes>=1が必要です')
    if not 0.3 <= args.width <= 2.0:
        p.error('widthは0.3〜2.0mで指定してください')
    if args.seconds <= 0 or args.stall_seconds <= 0:
        p.error('時間設定は正の数で指定してください')

    train_env = make_env(args.width, args.seconds, args.stall_seconds)
    monitor = None
    try:
        obs, _ = train_env.reset(seed=42)
        print('110m平均台 [-5m, 105m]、ゴール100m、5m刻みCP')
        print('観測次元:', obs.shape, '行動次元:', train_env.action_space.shape)
        assert obs.shape == (366,) and train_env.action_space.shape == (17,)
        if args.test:
            for _ in range(300):
                obs, _, done, trunc, info = train_env.step(
                    np.zeros(17, dtype=np.float32)
                )
                if done or trunc:
                    print('試験終了:', info['reason'], '最前部:', info['distance_m'])
                    break
            print('動作テストOK')
            return

        monitor = Monitor(
            train_env,
            filename=str(OUT / f'training_{datetime.now():%Y%m%d_%H%M%S}.monitor.csv'),
            info_keywords=('distance_m', 'is_success', 'reason', 'checkpoints_passed'),
        )
        latest = OUT / 'model_latest.zip'
        if latest.exists():
            model = PPO.load(str(latest), env=monitor, device='cpu')
            print('再開:', latest)
        elif args.init == 'beam1m':
            if not OLD_BEAM_MODEL.exists():
                raise FileNotFoundError(OLD_BEAM_MODEL)
            model = PPO.load(str(OLD_BEAM_MODEL), env=monitor,
                             device='cpu', learning_rate=3e-5)
            print('1mモデルを引き継ぎ:', OLD_BEAM_MODEL)
        else:
            if not STANDING_MODEL.exists():
                raise FileNotFoundError(STANDING_MODEL)
            # 359次元の直立方策を366次元に拡張して引き継ぐ
            model = previous.transfer_standing_policy(monitor)
            print('直立550万モデルから新規開始')

        model.learning_rate = 3e-5
        model.verbose = 0
        if args.eval_only:
            evaluate(model, args, 'manual')
            if args.video:
                record_video(model, args, 'manual')
            return

        if not (OUT / 'evaluation_summary.csv').exists():
            evaluate(model, args, 'initial')

        previous_steps = model.num_timesteps
        target_steps = previous_steps + args.steps
        try:
            while model.num_timesteps < target_steps:
                chunk = min(args.eval_every,
                            target_steps - model.num_timesteps)
                print(f'\n追加学習 {chunk:,} Step 開始...', flush=True)
                model.learn(total_timesteps=chunk,
                            reset_num_timesteps=False)
                mark = model.num_timesteps
                label = f'step_{mark}'
                model.save(str(OUT / f'{label}.zip'))
                model.save(str(latest))
                print('チェックポイント保存:', label)
                evaluate(model, args, label)
                if args.video:
                    record_video(model, args, label)
        except KeyboardInterrupt:
            model.save(str(OUT / 'model_interrupted.zip'))
            model.save(str(latest))
            print('\n中断モデルを保存しました。再開は同じコマンドでOK。')
            return
        except Exception:
            model.save(str(OUT / 'model_error.zip'))
            model.save(str(latest))
            raise
        print('\n学習完了。保存先:', OUT.resolve())
    finally:
        if monitor is not None:
            monitor.close()
        else:
            train_env.close()


if __name__ == '__main__':
    main()
