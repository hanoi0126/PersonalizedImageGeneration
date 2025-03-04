# PersonalizedImageGeneration

## 概要
- [FastComposer](https://github.com/mit-han-lab/fastcomposer) をベースに追加学習を行い、「複製効果」を抑制するための手法を提案する
- 複製効果を判定するための評価機構の提供


## Setup


### Local でのセットアップ方法

1. リポジトリをクローン
```bash
$ git clone git@github.com:hanoi0126/PersonalizedImageGeneration.git
$ cd PersonalizedImageGeneration
```

2. Docker の起動
- `run.sh` には `-g` / `-a` のどちらかを渡す必要があり、`-g` には使用する GPU のインデックス番号も渡す
```bash
$ bash scripts/docker_build.sh  # docker build
$ bash scripts/docker_run.sh -g 0  # docker run
```

3. Docker にアタッチして環境に入り、ライブラリのインストール
```bash
root@xxxxxxxxxxxx: poetry install --no-root
```

4. Model のダウンロード
- Base Model: [StableDiffusion v1.5](https://huggingface.co/jyoung105/stable-diffusion-v1-5/tree/main) 
- Pre-trained Model: [mit-han-lab/fastcomposer](https://huggingface.co/mit-han-lab/fastcomposer)
```bash
root@xxxx: bash scripts/download_model.sh  # install stable diffusion
```

5. 学習・推論・評価
```bash
root@xxxx: poetry run python fastcomposer/train.py  # 学習
root@xxxx: poetry run python fastcomposer/infer.py  # 推論
root@xxxx: poetry run python fastcomposer/evaluate.py  # 評価
```


### Wisteria 環境でのセットアップ方法

1. リポジトリをクローン
```bash
$ git clone git@github.com:hanoi0126/PersonalizedImageGeneration.git
$ cd PersonalizedImageGeneration
```

2. Model のダウンロード
```bash
$ pjsub jobs/download.sh  # ダウンロードジョブを投入
```

3. 学習・推論
- 学習の場合
```bash
$ pjsub jobs/run.sh  # 学習ジョブを投入
[INFO] PJM 0000 pjsub Job 01234 submitted.
```

- 推論の場合
```bash
$ pjsub jobs/run_infer.sh  # 推論ジョブを投入
[INFO] PJM 0000 pjsub Job 56789 submitted.
```



