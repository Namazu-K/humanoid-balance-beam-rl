"""Six-phase curriculum. Default is a plan; --run explicitly starts training."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MUJOCO_GL','osmesa')
import sys
sys.modules['triton']=None
import argparse,csv,hashlib,importlib,json,math,shutil,signal,subprocess,time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
LIB=ROOT/'envs'
STOP=False

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def stopped(stop_file):return STOP or Path(stop_file).exists()
def handle_stop(*_):
    global STOP
    STOP=True

def settings(stage,out,stop_file,smoke=False,envs_override=None):
    d=dict(stage.get('params',{}))
    d.update(out=str(out),stop_file=str(stop_file),seed=stage['seed'],seconds=stage['seconds'],
             envs=stage['envs'],rollout_total=2048,resume=False)
    if stage['environment']!='flat_standing' and envs_override is not None:d['envs']=envs_override
    if smoke:
        d.update(seconds=1.2,rollout_total=128)
        if stage['environment']!='flat_standing':d['envs']=2
    return SimpleNamespace(**d)

def make_environment(stage,out,stop_file,smoke=False,envs_override=None):
    sys.path.insert(0,str(LIB)) if str(LIB) not in sys.path else None
    os.chdir(out)
    import torch
    torch.set_num_threads(1)
    backend=importlib.import_module(stage['backend'])
    args=settings(stage,out,stop_file,smoke,envs_override)
    if stage['environment']=='flat_standing':
        env=backend.make_flat_env() if stage['id'].startswith('01') else backend.make_env()
        if smoke:env.episode_limit=1.2
        return env
    if stage['environment']=='beam':
        if not (out/'beam.xml').exists():backend.build_beam(out/'beam.xml',width=1.)
    else:
        import train_flat100_groundfix_camera as ground
        backend.GROUND_XML=out/'flat_ground_solid.xml'
        if not backend.GROUND_XML.exists():
            ground.runner.OUT=out;ground.solid_ground_xml(1.)
    runner=backend.configure_existing_env(out,args)
    return runner.make_env(1.,args.seconds,args.stall_window)

def factory(stage,out,stop_file,smoke,envs_override,index):
    def create():
        from stable_baselines3.common.monitor import Monitor
        env=make_environment(stage,out,stop_file,smoke,envs_override)
        return Monitor(env,filename=str(out/f'worker_{index}.monitor.csv'),override_existing=False)
    return create

def evaluate(model,stage,out,stop_file,smoke,envs_override):
    import numpy as np
    env=make_environment(stage,out,stop_file,smoke,envs_override)
    count=2 if smoke else stage['evaluation_episodes']
    rows=[]
    try:
        for i in range(count):
            if stopped(stop_file):raise InterruptedError('STOP requested')
            beam=stage['environment']=='beam'
            seed=6000+i//3 if beam else 8000+i
            options={'lateral_offset':(-.3,0.,.3)[i%3]} if beam else None
            obs,_=env.reset(seed=seed,options=options)
            reward_sum=0.
            while True:
                if stopped(stop_file):raise InterruptedError('STOP requested')
                action,_=model.predict(obs,deterministic=True)
                obs,reward,done,trunc,info=env.step(action);reward_sum+=float(reward)
                if done or trunc:break
            rows.append(dict(steps=model.num_timesteps,seed=seed,offset=options['lateral_offset'] if beam else '',
                seconds=float(info.get('elapsed_seconds',0)),reward=reward_sum,reason=info.get('reason',''),
                distance_m=float(info.get('net_forward_m',0)),
                strict_stable_s=float(info.get('longest_stable_seconds',0)),
                relaxed_stable_s=float(info.get('longest_relaxed_seconds',0)),
                lateral_m=float(info.get('mean_abs_lateral_m',0))))
        path=out/'evaluation_episodes.csv';exists=path.exists()
        with path.open('a',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));
            if not exists:w.writeheader()
            w.writerows(rows)
        return dict(episodes=len(rows),mean_distance_m=float(np.mean([r['distance_m'] for r in rows])),
                    mean_stable_s=float(np.mean([r['strict_stable_s'] for r in rows])),
                    mean_relaxed_s=float(np.mean([r['relaxed_stable_s'] for r in rows])))
    finally:env.close()

def worker(cli):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
    torch.set_num_threads(1)
    stage=json.loads((ROOT/'curriculum.json').read_text(encoding='utf-8'))['stages'][cli.worker]
    out=Path(cli.out).resolve();out.mkdir(parents=True,exist_ok=True)
    stop_file=out.parent/'STOP'
    shutil.copy2(ROOT/'humanoid_flat_feet_heel.xml',out/'humanoid_flat_feet_heel.xml')
    args=settings(stage,out,stop_file,cli.smoke,cli.envs)
    signal.signal(signal.SIGINT,handle_stop);signal.signal(signal.SIGTERM,handle_stop)
    probe=make_environment(stage,out,stop_file,cli.smoke,cli.envs)
    observation_shape=probe.observation_space.shape
    expected_shape=(359,) if stage['environment']=='flat_standing' else (366,)
    assert observation_shape==expected_shape,(observation_shape,expected_shape)
    probe.close()
    functions=[factory(stage,out,stop_file,cli.smoke,cli.envs,i) for i in range(args.envs)]
    vec=DummyVecEnv(functions) if args.envs==1 else SubprocVecEnv(functions,start_method='spawn')
    model=None
    class StopCallback(BaseCallback):
        def _on_step(self):return not stopped(stop_file)
    callback=StopCallback()
    latest=out/'model_latest.zip'
    continuing=cli.resume and latest.exists()
    source=latest if continuing else Path(cli.model).resolve() if cli.model else None
    try:
        if stage['environment']=='flat_standing':
            if source is None:
                model=PPO('MlpPolicy',vec,learning_rate=stage['learning_rate'],n_steps=128 if cli.smoke else 2048,
                    batch_size=64,n_epochs=1 if cli.smoke else 4,gamma=.999,gae_lambda=.95,clip_range=.15,
                    target_kl=.02,ent_coef=0.,policy_kwargs={'log_std_init':-2.},seed=stage['seed'],device='cpu',verbose=0)
            else:
                model=PPO.load(str(source),env=vec,device='cpu',learning_rate=stage['learning_rate'],gamma=.999,target_kl=.02,
                               n_steps=128 if cli.smoke else 2048,verbose=0)
            transfer='standing_359'
        else:
            if source is None:raise ValueError('Walking/beam phases require an input model')
            backend=importlib.import_module(stage['backend'])
            model,transfer=backend.load_start_model(source,vec,args)
        if cli.smoke:model.n_epochs=1
        identity=out/'phase_state.json'
        previous=json.loads(identity.read_text()) if continuing and identity.exists() else None
        start=previous['start_steps'] if previous else model.num_timesteps
        target=previous['target_steps'] if previous else start+128 if cli.smoke else stage['target']
        if model.num_timesteps>target:raise ValueError('Input checkpoint is beyond this phase target')
        write_json(identity,dict(stage=stage,start_steps=start,target_steps=target,source=str(source),transfer=transfer,
                   observation_shape=observation_shape,envs=args.envs,rollout_total=args.rollout_total,smoke=cli.smoke))
        # Fixed target checkpoints replace the original manual choice of intermediate models.
        if cli.smoke:segments=[dict(target=target,eval_every=128)]
        elif 'milestones' in stage:segments=[dict(target=n,eval_every=n-start) for n in stage['milestones'] if n>start]
        elif 'segments' in stage:segments=stage['segments']
        else:segments=[dict(target=target,eval_every=stage['eval_every'])]
        segment_start=start
        for seg in segments:
            end=min(int(seg['target']),target)
            trained_in_segment=False
            while model.num_timesteps<end:
                if stopped(stop_file):raise InterruptedError('STOP requested')
                interval=max(1,int(seg['eval_every']))
                mark=min(end,segment_start+((model.num_timesteps-segment_start)//interval+1)*interval)
                model.learn(total_timesteps=mark-model.num_timesteps,reset_num_timesteps=False,callback=callback)
                trained_in_segment=True
                model.save(str(latest))
                if stopped(stop_file):raise InterruptedError('STOP requested')
                checkpoint=out/f'step_{model.num_timesteps}.zip';model.save(str(checkpoint))
                score=evaluate(model,stage,out,stop_file,cli.smoke,cli.envs)
                write_json(out/'status.json',dict(stage='training',steps=model.num_timesteps,target=target,evaluation=score))
                print(f"{stage['id']} | {model.num_timesteps:,}/{target:,} | {score}",flush=True)
            segment_start=end
            reload_now=end in stage.get('reload_at',[]) or ('segments' in stage and end<target)
            if reload_now and trained_in_segment and not cli.smoke:
                if stage['environment']=='flat_standing':model=PPO.load(str(latest),env=vec,device='cpu',verbose=0)
                else:model,_=backend.load_start_model(latest,vec,args)
        selected=out/'selected_model.zip';model.save(str(selected))
        write_json(out/'completed.json',dict(phase=stage['id'],steps=model.num_timesteps,target_steps=target,
            model=str(selected),sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),smoke=cli.smoke))
        return 0
    except InterruptedError:
        if model is not None:model.save(str(latest))
        write_json(out/'status.json',dict(stage='stopped',steps=model.num_timesteps if model else None))
        return 3
    finally:vec.close()

def main():
    parser=argparse.ArgumentParser(description='平地立位→平地歩行→幅1m平均台。--runを付けたときだけ長時間学習。')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='短い実学習で全段階の接続だけを確認')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--start-phase',type=int,default=1,choices=range(1,7))
    parser.add_argument('--end-phase',type=int,default=6,choices=range(1,7))
    parser.add_argument('--model',type=Path,help='途中段階から開始するときのモデル')
    parser.add_argument('--envs',type=int,choices=range(1,17),help='歩行以降の並列数。既定16、立位は1')
    parser.add_argument('--worker',type=int,help=argparse.SUPPRESS)
    cli=parser.parse_args()
    if cli.worker is not None:return worker(cli)
    config=json.loads((ROOT/'curriculum.json').read_text(encoding='utf-8'))
    if not cli.run and not cli.smoke:
        for s in config['stages']:print(f"{s['id']}: {s['title']} → 累積{s['target']:,} | {s['reward']}")
        print('実行例: python train_curriculum.py --run --out results/my_curriculum')
        return 0
    if cli.out is None:parser.error('--outが必要です')
    if cli.start_phase>cli.end_phase:parser.error('start-phaseはend-phase以下にしてください')
    if cli.start_phase>1 and cli.model is None and not cli.resume:parser.error('途中開始には--modelが必要です')
    out=cli.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if cli.model is not None and not cli.model.resolve().is_file():parser.error('開始モデルがありません')
    fingerprint=hashlib.sha256(json.dumps(dict(config=config,envs=cli.envs,smoke=cli.smoke),sort_keys=True).encode()).hexdigest()
    meta=out/'pipeline.json'
    if meta.exists():
        if not cli.resume:parser.error('既存出力です。継続する場合は--resumeを指定してください')
        if json.loads(meta.read_text())['fingerprint']!=fingerprint:parser.error('再開時の設定が変わっています')
    elif any(out.iterdir()):parser.error('新規実行には空の保存先を指定してください')
    else:write_json(meta,dict(fingerprint=fingerprint,smoke=cli.smoke,config=config))
    model=cli.model.resolve() if cli.model else None
    for i in range(cli.start_phase-1,cli.end_phase):
        phase=out/config['stages'][i]['id'];completed=phase/'completed.json'
        if stopped(out/'STOP'):
            print('STOPファイルがあります。再開前に取り除いてください。');return 3
        if completed.exists():
            saved=json.loads(completed.read_text());model=Path(saved['model'])
            if not model.exists() or hashlib.sha256(model.read_bytes()).hexdigest()!=saved['sha256']:raise RuntimeError('保存モデルの照合に失敗')
            continue
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(i),'--out',str(phase)]
        if model is not None:command+=['--model',str(model)]
        if cli.smoke:command+=['--smoke']
        if cli.resume:command+=['--resume']
        if cli.envs is not None:command+=['--envs',str(cli.envs)]
        result=subprocess.run(command)
        if result.returncode:return result.returncode
        model=Path(json.loads(completed.read_text())['model'])
    write_json(out/'completed.json',dict(model=str(model),end_phase=cli.end_phase,smoke=cli.smoke,
        note='Smoke test weights are not trained demonstration models.' if cli.smoke else 'New training run; performance is not guaranteed to match checkpoint 92508160.'))
    print('完了:',model,flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
