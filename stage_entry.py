"""Numbered entry points share the same curriculum implementation."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run_stage(number):
    parser=argparse.ArgumentParser(description=f'学習段階{number}のみを実行。--runなしでは設定表示だけ。')
    parser.add_argument('--out',type=Path,help='6段階で共通の保存先')
    parser.add_argument('--model',type=Path,help='省略時は前段階のselected_model.zip')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--envs',type=int,choices=range(1,17))
    args=parser.parse_args()
    stages=json.loads((ROOT/'curriculum.json').read_text(encoding='utf-8'))['stages']
    current=stages[number-1]
    if not args.run and not args.smoke:
        print(f"{current['id']}：{current['title']}")
        print(f"環境：{current['environment']} / 目標累積：{current['target']:,}")
        print('報酬：'+current['reward'])
        print('学習する場合は --run --out results/my_curriculum を指定してください。')
        return 0
    if args.out is None:parser.error('--outが必要です。全段階で同じ保存先を指定してください')
    out=args.out.resolve()
    model=args.model.resolve() if args.model else None
    if number>1 and model is None:
        model=out/stages[number-2]['id']/'selected_model.zip'
        if not model.is_file():parser.error('前段階のモデルがありません。前段階を実行するか、--modelで指定してください')
    command=[sys.executable,'-u',str(ROOT/'train_curriculum.py'),'--start-phase',str(number),'--end-phase',str(number),'--out',str(out)]
    if args.run:command.append('--run')
    if args.smoke:command.append('--smoke')
    if args.resume or (out/'pipeline.json').exists():command.append('--resume')
    if model is not None:command+=['--model',str(model)]
    if args.envs is not None:command+=['--envs',str(args.envs)]
    return subprocess.run(command).returncode
