# 平地で立つ → 平地で歩く → 幅1m平均台：連続学習用

**各段階で採用した学習済みモデル6個を`models/`に同梱しています。提出動画のモデルは`models/06_beam_gaussian_92508160.zip`です。** 過去に学習した重みであり、この整理版で新たに学習したテスト重みではありません。

## 学習済みモデルからすぐ評価する

学習し直さずに幅1m平均台で動かす場合は、展開したこのフォルダで次を実行する。

```bash
MUJOCO_GL=osmesa /root/rope/.venv/bin/python evaluate_pretrained.py
```

既定は92508160・seed9040・中央スタート・最大600秒。最良動画の初期条件で、学習は行わない。結果は`results/evaluate_92508160/result.json`へ保存する。再実行時は空の`--out`を指定する。

読み込みだけを先に確認する場合：

```bash
/root/rope/.venv/bin/python evaluate_pretrained.py --check --out results/model_check
```

|段階|同梱モデル|用途|
|---|---|---|
|01|`models/01_stand_initial_2000896.zip`|初期の平足立位|
|02|`models/02_stand_relaxed_5500928.zip`|歩行の起点となった立位|
|03|`models/03_flat_walk_34502656.zip`|平地での歩行獲得|
|04|`models/04_flat_speed_39503872.zip`|平地での速度調整後|
|05|`models/05_flat_log_64507904.zip`|平均台学習の起点|
|06|`models/06_beam_gaussian_92508160.zip`|提出動画に採用した平均台モデル|

モデルの実カウンターとSHA-256は`pretrained_models.json`に記載。**学習済みモデルのZIPは展開しない。** Stable-Baselines3がそのまま読み込む。

途中学習を始める際にも`--model models/05_flat_log_64507904.zip`のように利用できる。ただし92508160は06の終了目標そのものなので、それより先の追加学習を行うには設定の終了目標も変更する必要がある。

GitHubへは、このフォルダの中身を`models/`ごとアップロードする。リポジトリのトップ表示用に`README.md`も同梱する。実行時の`results/`、仮想環境、動画は含めない。

## この一式の目的

92508160へ至った学習環境・報酬の変更だけを残し、順番に実行できる形へ整理した。一括実行の入口は**`train_curriculum.py`**。段階の設定は`curriculum.json`にまとめ、段階別の番号付きファイルからも同じ共通処理を呼ぶ。同じ報酬の継続を、単に保存先や学習量が違うだけで別プログラムへ分けない。

`envs/`の13本は、元の環境・報酬・観測変換を再利用する共通実装。ファイル名に`train_`が付くものもあるが、利用者がそれらを順番に起動する必要はない。比較枝の学習を実行する構成でもない。

段階ごとの入口も番号付きで用意した。**一括実行と段階別実行は同じ学習処理を呼ぶ**ので、報酬や設定を二重管理しない。

|段階|段階別の実行ファイル|環境|引き継ぎ後の累積目標|
|---|---|---|---:|
|01|`01_train_stand_initial.py`|平地・初期立位|2,000,896|
|02|`02_train_stand_relaxed.py`|同じ平地・立位報酬変更|5,500,928|
|03|`03_train_flat_walk.py`|広い平地・歩行|34,502,656|
|04|`04_train_flat_speed.py`|同じ平地・速度調整|39,503,872|
|05|`05_train_flat_log.py`|同じ平地・対数速度報酬|64,507,904|
|06|`06_train_beam_gaussian.py`|幅1m・長さ100m平均台|92,508,160|

01から順に、以下のように**同じ`--out`**を指定して実行できる。02以降は、前段階の`selected_model.zip`を自動で読み込む。各行は前の行が完了してから実行する。

```bash
/root/rope/.venv/bin/python -u 01_train_stand_initial.py --run --out results/my_curriculum
/root/rope/.venv/bin/python -u 02_train_stand_relaxed.py --run --out results/my_curriculum
/root/rope/.venv/bin/python -u 03_train_flat_walk.py --run --out results/my_curriculum
/root/rope/.venv/bin/python -u 04_train_flat_speed.py --run --out results/my_curriculum
/root/rope/.venv/bin/python -u 05_train_flat_log.py --run --out results/my_curriculum
/root/rope/.venv/bin/python -u 06_train_beam_gaussian.py --run --out results/my_curriculum
```

保存先が既存なら整合性を確認して再開する。並列数を指定する場合は各段階で同じ`--envs`を指定する。元モデルから始める段階は`--model`で指定可能。`--run`なしでは、その段階の設定を表示するだけで学習しない。

## 学習環境と段階

|段階|物理的な学習環境|内容と報酬の違い|目安の累積ステップ|
|---|---|---|---:|
|01 立位・初期|**平地**。平足Humanoid、移動目標なし|胴体の姿勢・高さ・接地・支持重心を評価する初期立位報酬|0 → 2,000,896|
|02 立位・改良|**同じ平地・同じ平足**|45度・高さ低下38cmまでを緩和安定条件とし、姿勢・支持を評価。操作量や急変等を減点|2,000,896 → 5,500,928|
|03 歩行|**広い平地**。平均台の幅制限なし|10mゴールへの接近、0.35m/sの速度追跡、姿勢・横ずれ等を評価。観測359→366次元へ拡張|5,500,928 → 34,502,656|
|04 速度調整|**同じ広い平地**|目標0.45m/s、前進・速度の重み等を変更|34,502,656 → 39,503,872|
|05 対数速度|**同じ広い平地**|ゴールへ近づく速度の符号付き対数報酬。重み2・尺度0.25、落下罰10等|39,503,872 → 64,507,904|
|06 平均台|**幅1m・長さ100m**。下の地面は高さ−1.5m|骨盤が新しく進んだ区間のかまぼこ高さを積分。転倒・落下−10。CLFを報酬・制御に使わない|64,507,904 → 92,508,160|

ロボットは全段階で平足（24×14×5cm、足の中心を前方へ5cm）を使う。丸足で立つ初期の探索や、腹ばいの学習は、この重みの採用経路には入れない。

01と02は同じ平地だが報酬が変わるので分ける。**02の200万→300万→550万は同じ環境・報酬なので1段階へ統合**した。05も、100万・500万・1000万・途中の900万という実行の区切りを、同じ対数報酬の1段階にまとめた。

途中の歩行・速度段階には、姿勢等に関するCLF着想のペナルティが含まれるが、数学的な安定性保証はない。06の最終報酬とは区別する。

## 動かし方（WSL Ubuntu・CPU）

このフォルダを任意の場所に展開する。従来の`/root/rope`へファイルを混ぜる必要はない。出力は指定した`--out`に保存する。相対パスの基準は起動した場所。

既存の仮想環境を使う場合：

```bash
cd /root/rope/curriculum_92508160
/root/rope/.venv/bin/python train_curriculum.py
```

上は**計画の表示だけ**で、学習は始めない。新しく環境を作る場合はPythonの仮想環境を作り、`pip install -r requirements.txt`で依存関係をインストールする。記録した版が使用環境で入手可能かは各パッケージ配布元に依存する。

### 短い接続確認

```bash
MUJOCO_GL=osmesa /root/rope/.venv/bin/python train_curriculum.py \
  --smoke --out results/smoke
```

6段階を各128サンプル程度だけ学習して、保存・読み込み・359→366次元変換・平均台への切替を確認する。**合計768サンプルのテストであり、歩けるモデルを作る処理ではない。** 保存先も本学習と分ける。

### 0から最後まで連続学習

```bash
MUJOCO_GL=osmesa /root/rope/.venv/bin/python -u train_curriculum.py \
  --run --out results/retrain_92508160
```

01・02の立位は1環境、それ以降は既定16環境、PPO更新の標本数は合計2048。CPU負荷を下げる場合は`--envs 8`などを付けられるが、元の実行条件とは変わる。

### 立位までで止める

```bash
MUJOCO_GL=osmesa /root/rope/.venv/bin/python -u train_curriculum.py \
  --run --end-phase 2 --out results/standing_only
```

### 指定モデルから途中開始

例：元の550万立位モデルから歩行以降を実行する。

```bash
MUJOCO_GL=osmesa /root/rope/.venv/bin/python -u train_curriculum.py \
  --run --start-phase 3 \
  --model /root/rope/results/v5_to_10m/model_5500k.zip \
  --out results/from_standing_5500k
```

例：元の64507904から平均台だけを実行する。

```bash
MUJOCO_GL=osmesa /root/rope/.venv/bin/python -u train_curriculum.py \
  --run --start-phase 6 \
  --model /root/rope/results/current_log_continue_20m/training/step_64507904.zip \
  --out results/beam_only
```

入力モデルのカウンターは当該段階の終了目標以下である必要がある。立位は359次元、歩行以降は366次元を使用する。

### 中断・再開

保存先の直下に`STOP`を作るとモデルを保存して止まる。

```bash
touch results/retrain_92508160/STOP
```

再開前にその`STOP`だけを取り除き、同じ設定で`--resume`を付ける。

```bash
rm results/retrain_92508160/STOP
MUJOCO_GL=osmesa /root/rope/.venv/bin/python -u train_curriculum.py \
  --run --resume --out results/retrain_92508160
```

途中段階から始めた実行を再開する場合は、開始時と同じ`--start-phase`等も指定する。完了した段階はモデルのハッシュを照合してスキップし、進行中の段階は`model_latest.zip`から再開する。物理状態・乱数状態まで復元する厳密な同一軌跡の再開ではない。

## 保存されるもの

```text
指定した保存先/
  pipeline.json
  01_stand_initial/
  02_stand_relaxed/
  03_flat_walk/
  04_flat_speed/
  05_flat_log/
  06_beam_gaussian/
    phase_state.json
    worker_*.monitor.csv
    evaluation_episodes.csv
    step_*.zip
    model_latest.zip
    selected_model.zip
    completed.json
  completed.json
```

各段階の最後の`selected_model.zip`を次へ自動的に引き継ぐ。評価は立位100回、歩行20回、平均台21回で、各区切りに距離・立位時間等を保存する。今回の整理版では評価動画や関節の詳細解析を毎回生成しない。

## 元の経路との関係・再現の範囲

この構成は「近いものを目指して同じカリキュラムを再学習する」ための整理版。過去の92508160をビット単位で再現するプログラムではなく、同じ成績も保証しない。

- 元の途中モデル選択を、対応する固定の累積ステップでの引き継ぎに置き換えた。再実行時にもそこが最良とは限らないので、途中モデルも保存する。
- 元の学習で採用しなかった後続の学習は行わない。歩行は34502656相当、対数型は64507904相当、平均台は92508160相当までで止める。
- 保存先の変更だけでファイルを分けず、同じ環境・報酬は共通化した。一方、元の主要な環境・報酬実装は`envs/`から再利用した。
- 立位の300万、対数型の40505344・45506560・55506944相当では、元のスクリプト切替に近づけるため内部で保存・読み込みを挟む。同じ報酬なので利用者が別ファイルを起動する必要はない。
- 一部の初期実行引数や全過去時点のソースが残っておらず、ライブラリやCPU、乱数、並列実行でも結果は変わる。

短い接続テストは6段階すべて完了した。統合前の対数型v7と、採用したv8のlog分岐について、同じ初期条件・同じ行動を300ステップ与え、全観測と報酬が一致することも確認した。これは接続・報酬の整合性確認であり、長時間再学習の成績確認ではない。今回、新たな9250万ステップの本学習は実行していない。

同じ報酬の2区間を続ける保存・再読み込みも、4096サンプルだけの専用設定で確認済み。検証結果は`verification/`に収録し、テスト用の未学習に近い重みは配布物に含めていない。

## 出典・ライセンス

Humanoid XMLはGymnasium付属モデルの変更版で、平足化・初期高さを変更した。GymnasiumのMIT原文は`LICENSES/gymnasium-LICENSE`。依存ライブラリ等の参照用のライセンス文書も保持する。ライブラリの本体・FFmpeg実行ファイルは収録しない。プロジェクト独自部分に、この整理作業で新たなライセンスを付与していない。
