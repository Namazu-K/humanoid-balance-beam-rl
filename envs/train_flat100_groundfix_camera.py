"""Flat-ground locomotion: independent collision box + reset diagnostics.

Place next to train_flat100_nofall.py and train_beam110.py in ~/rope.

Test: MUJOCO_GL=osmesa python -u train_flat100_groundfix.py --diagnose
Train: MUJOCO_GL=osmesa python -u train_flat100_groundfix.py --init standing \
    --seconds 40 --steps 200000 --eval-every 50000 --episodes 20 --video
"""
import sys
sys.modules["triton"] = None

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

import train_flat100_nofall as previous

runner = previous.runner

# Never accidentally reuse weights from the possibly invalid ground experiment.
runner.OUT = Path("results/flat100_groundfix")
runner.OUT.mkdir(parents=True, exist_ok=True)


def solid_ground_xml(_width):
    """Build a known-collidable box, with its TOP at z=0.

    This deliberately creates a new ground geom instead of changing the
    existing floor's type; inherited XML attributes cannot leak through.
    The broad box spans x=-200..300 and y=-100..100 meters.
    """
    tree = ET.parse(runner.XML_SOURCE)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("worldbodyが見つかりません")

    # Remove original world floor and any previous lower ground.
    for geom in list(world.findall("geom")):
        if geom.get("name") in ("floor", "ground_below"):
            world.remove(geom)

    ET.SubElement(world, "geom", {
        "name": "floor", "type": "box",
        "pos": "50 0 -0.20", "size": "250 100 0.20",
        "contype": "1", "conaffinity": "1", "condim": "3",
        "friction": "1.5 0.1 0.1",
        "rgba": "0.57 0.62 0.68 1",
    })
    # Compatibility: wrapper expects a geom named ground_below.
    # Place it very far away, with contacts disabled.
    ET.SubElement(world, "geom", {
        "name": "ground_below", "type": "box",
        "pos": "0 0 -1002", "size": "1 1 1",
        "contype": "0", "conaffinity": "0", "rgba": "0 0 0 0",
    })

    # Visual-only metre markers: physically noncollidable.
    for x in range(0, 101, 5):
        color = ("0.1 0.8 0.2 1" if x == 100 else
                 "0.15 0.4 0.9 1" if x == 0 else
                 "0.95 0.82 0.18 1")
        ET.SubElement(world, "geom", {
            "name": f"distance_mark_{x}m", "type": "box",
            "pos": f"{x} 0 0.003", "size": "0.035 1.0 0.002",
            "contype": "0", "conaffinity": "0", "rgba": color,
        })

    out = (runner.OUT / "flat_ground_solid.xml").resolve()
    tree.write(out, encoding="unicode")
    return out


# Original Flat100Env reward / 366 observation / evaluate all retained.
# Only replace the physical terrain and output directory.
runner.beam_xml = solid_ground_xml


def diagnose():
    env = runner.make_env(1.0, 40.0, 30.0)
    try:
        obs, _ = env.reset(seed=42)
        model, data = env.base.model, env.base.data
        floor_id = model.geom("floor").id
        g = model.geom("floor")
        print("\n=== 平地と初期位置の診断 ===", flush=True)
        print("観測:", obs.shape, "アクチュエータ:", model.nu)
        print("床の種類: box / 上端z=0m")
        print("床の接触属性 contype:", int(g.contype),
              "conaffinity:", int(g.conaffinity))
        print("開始時の胴体z:", round(float(data.body("torso").xpos[2]), 3), "m")
        print("開始時のルートz:", round(float(data.qpos[2]), 3), "m")
        for side in ("left", "right"):
            foot = data.geom(f"{side}_foot")
            print(f"{side}足ジオム中心z:", round(float(foot.xpos[2]), 3), "m")

        touched = 0
        lowest_torso = float('inf')
        for i in range(135):
            act = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, _, done, trunc, info = env.step(act)
            contact = False
            for j in range(data.ncon):
                c = data.contact[j]
                if int(c.geom1) == floor_id or int(c.geom2) == floor_id:
                    contact = True
                    break
            touched += int(contact)
            torso_z = float(data.body("torso").xpos[2])
            lowest_torso = min(lowest_torso, torso_z)
            if i in (0, 16, 33, 66, 100, 134):
                print(f"t={info['elapsed_seconds']:.2f}s "
                      f"胴体z={torso_z:.3f}m "
                      f"床接触={contact} "
                      f"旧転倒判定={info.get('old_reason', '?')}", flush=True)
            if done or trunc:
                print("終了:", info.get("reason"))
                break

        print("床と接触した計測ステップ:", touched)
        print("最小胴体z:", round(lowest_torso, 3), "m")
        if touched == 0 or lowest_torso < -0.5:
            print("【異常】床の当たり判定または初期位置に問題があります。学習せず結果を送ってください。")
        else:
            print("【OK】床との衝突が確認できました。動画で見え方も確認してください。")
    finally:
        env.close()


def record_video_with_safe_camera(model, args, label, seed=5000):
    """Use a camera that never tracks below floor level."""
    import mujoco
    import imageio.v2 as imageio

    env = runner.make_env(args.width, args.seconds, args.stall_seconds)
    renderer = writer = None
    path = runner.OUT / f"{label}_seed{seed}.mp4"
    try:
        obs, _ = env.reset(seed=seed)
        # Huge terrain and dummy geometry can inflate stat.extent.
        # MuJoCo near/far clipping = vis.map.{znear,zfar} * stat.extent.
        m = env.base.model
        before = float(m.stat.extent * m.vis.map.znear)
        print(f"Camera near BEFORE: {before:.2f}m; "
              f"extent={float(m.stat.extent):.1f}m", flush=True)
        m.stat.extent = 3.0
        m.vis.map.znear = 0.01
        m.vis.map.zfar = 200.0
        print(f"Camera near AFTER: {float(m.stat.extent * m.vis.map.znear):.3f}m; "
              f"far={float(m.stat.extent * m.vis.map.zfar):.1f}m", flush=True)
        renderer = mujoco.Renderer(env.base.model, width=640, height=480)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 4.8
        camera.azimuth = 135
        camera.elevation = -15
        writer = imageio.get_writer(str(path), fps=20, codec="libx264", quality=8)

        def capture():
            pos = env.base.data.body("torso").xpos
            camera.lookat[:] = np.array([pos[0], pos[1], max(0.8, float(pos[2]) - 0.3)])
            renderer.update_scene(env.base.data, camera=camera)
            writer.append_data(np.array(renderer.render(), copy=True))

        capture()
        next_frame = 0.05
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(action)
            if float(env.base.elapsed) >= next_frame:
                capture()
                next_frame += 0.05
            if done or trunc:
                break
        print("録画:", path, "最大先端前進:", f"{info['distance_m']:.3f}m")
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()


runner.record_video = record_video_with_safe_camera


if __name__ == "__main__":
    if "--diagnose" in sys.argv:
        diagnose()
    else:
        print("【新実験】平地の衝突判定を新規作成・独立保存先", flush=True)
        runner.main()
