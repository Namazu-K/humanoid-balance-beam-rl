"""Run the historical beam policy without training."""
import argparse,hashlib,json,time
from pathlib import Path
import train_curriculum as curriculum

def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description='92508160モデルで幅1m平均台を評価。学習はしません。')
    parser.add_argument('--model',type=Path,default=root/'models/06_beam_gaussian_92508160.zip')
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--seed',type=int,default=9040)
    parser.add_argument('--offset',type=float,default=0.)
    parser.add_argument('--out',type=Path,default=Path('results/evaluate_92508160'))
    parser.add_argument('--check',action='store_true',help='モデル・環境を読み込み、形状と初回推論だけ確認')
    args=parser.parse_args()
    if args.seconds<=0:parser.error('--secondsは正の数を指定してください')
    source=args.model.resolve()
    if not source.is_file():parser.error('モデルがありません')
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):parser.error('空の保存先を指定してください')
    out.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(root/'humanoid_flat_feet_heel.xml',out/'humanoid_flat_feet_heel.xml')
    config=json.loads((root/'curriculum.json').read_text(encoding='utf-8'))
    stage=config['stages'][5];stage['seconds']=args.seconds
    env=curriculum.make_environment(stage,out,out/'STOP')
    from stable_baselines3 import PPO
    import numpy as np
    model=PPO.load(str(source),device='cpu');model.policy.set_training_mode(False)
    try:
        assert env.observation_space.shape==model.observation_space.shape==(366,)
        assert env.action_space.shape==model.action_space.shape==(17,)
        obs,_=env.reset(seed=args.seed,options={'lateral_offset':args.offset})
        action,_=model.predict(obs,deterministic=True)
        if args.check:
            result=dict(model_steps=model.num_timesteps,observation_shape=list(obs.shape),action_shape=list(action.shape),checked=True)
        else:
            digest=hashlib.sha256();steps=0
            while True:
                if (out/'STOP').exists():raise InterruptedError('STOP requested')
                action,_=model.predict(obs,deterministic=True)
                digest.update(np.asarray(action,dtype=np.float32).tobytes())
                obs,_,done,trunc,info=env.step(action);steps+=1
                if done or trunc:break
            result=dict(model_steps=model.num_timesteps,seed=args.seed,start_offset=args.offset,
                seconds=float(info['elapsed_seconds']),distance_m=float(info['net_forward_m']),
                reason=info['reason'],simulation_steps=steps,action_sha256=digest.hexdigest(),training=False)
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:env.close()
if __name__=='__main__':main()
