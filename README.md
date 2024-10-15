# README

## Setup

1. Download Carla 0.9.10.1 and Carla 0.9.14 to somewhere outside of this repository. Create soft links to the simulators under this repository.

    ```shell
    ln -s /path/to/carla-simulator-0.9.10.1 ./CARLA_Leaderboard_10
    ln -s /path/to/carla-simulator-0.9.14 ./CARLA_Leaderboard_20
    ```

2. Pull submodules for the leaderboard.

    ```shell
    git submodule update --init --recursive
    ```

3. Install the required Python packages.

    ```shell
    conda env create -f environment.yml
    conda activate garage
    ```

## For CARLA Leaderboard 1.0

### Pre-trained Models

`autonomousvision/carla_garage` has provide a set of pretrained models [here](https://s3.eu-central-1.amazonaws.com/avg-projects-2/jaeger2023arxiv/models/pretrained_models.zip). The models are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0). These are the final model weights used in the paper, the folder indicates the benchmark. For the training and validation towns, we provide 3 models which correspond to 3 different training seeds. The format is `approach_trainingsetting_seed`. Each folder has an `args.txt` containing the training settings in text, a `config.pickle` containing all hyperparameters for the code and a `model_0030.pth` containing the model weights. Additionally, there are training logs for most models.

We assume you have downloaded the pretrained models and extracted them to `./ckpt`. The folder structure should look like this:

```shell
./ckpt
├── lav
│   ├── aim_02_05_withheld_0
│   ├── aim_02_05_withheld_1
│   ├── aim_02_05_withheld_2
│   ├── tfpp_02_05_withheld_0
│   ├── tfpp_02_05_withheld_1
│   └── tfpp_02_05_withheld_2
├── leaderboard
│   └── tfpp_wp_all_0
└── longest6
    ├── tfpp_all_0
    ├── tfpp_all_1
    └── tfpp_all_2
```

### Evaluation

To evaluate a model, you need to start a CARLA server:

```shell
cd ./CARLA_Leaderboard_10
./CarlaUE4.sh --port=2000 -opengl
```

Afterward, run the evaluation script:

```shell
cd scripts
./L10_run_evaluation.sh
```

We have provided our evaluation result with `ckpt/lav/tfpp_02_05_withheld_0` under Carla leaderboard 1.0's offline testing `leaderboard_10/leaderboard/data/routes_testing.xml` in `logs/L10_testing`.

## For CARLA Leaderboard 2.0
