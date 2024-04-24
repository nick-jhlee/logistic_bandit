Code for the paper [_Improved Regret Bounds of (Multinomial) Logistic Bandits via Regret-to-Confidence-Set Conversion_](https://arxiv.org/abs/2310.18554) (AISTATS 2024), by [Junghyun Lee](https://nick-jhlee.github.io/), [Se-Young Yun](https://fbsqkd.github.io/), and [Kwang-Sung Jun](https://kwangsungjun.github.io).

This is forked from https://github.com/criteo-research/logistic_bandit.

If you plan to use this repository or cite our paper, please use the following bibtex format:

```latex

@InProceedings{lee2024logistic,
  title = 	 {{Improved Regret Bounds of (Multinomial) Logistic Bandits via Regret-to-Confidence-Set Conversion}},
  author =       {Lee, Junghyun and Yun, Se-Young and Jun, Kwang-Sung},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {4474--4482},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://arxiv.org/pdf/2310.18554.pdf},
  url = 	 {https://arxiv.org/abs/2310.18554},
}

```

# Install
Clone the repository and run:
```shell
$ pip install .
```
Every time one changes the source code, one should run `pip install .` again to update the package.

# Usage

This code implements the `OFULog+` algorithm of [Lee et al. 2024](https://arxiv.org/abs/2310.18554), along with several baselines (oldest to newest):

- `GLM-UCB` from [Filippi et al. 2010](https://papers.nips.cc/paper/2010/file/c2626d850c80ea07e7511bbae4c76f4b-Paper.pdf),
- `OL2M` from [Zhang et al. 2016](http://proceedings.mlr.press/v48/zhangb16.pdf),
- `GLOC` from [Jun et al. 2017](https://proceedings.neurips.cc/paper/2017/file/28dd2c7955ce926456240b2ff0100bde-Paper.pdf),
- `LogUCB1` from [Faury et al. 2020](http://proceedings.mlr.press/v119/faury20a/faury20a.pdf),
- `OFULog-r` from [Abeille et al. 2021](http://proceedings.mlr.press/v130/abeille21a/abeille21a.pdf).
- `ada-OFU-ECOLog` from [Faury et al. 2022](https://proceedings.mlr.press/v151/faury22a.html)
- `ada-OFU-ECOLog-TS` from [Faury et al. 2022](https://proceedings.mlr.press/v151/faury22a.html)


Experiments can be run for several Logistic Bandit (_i.e_ structured Bernoulli feedback) environments, such as static and time-varying finite arm-sets, or inifinite arm-sets (_e.g._ unit ball).
Note that the Thompson Sampling type algorithm (TS) is only available for `GLOC`, `OL2M`, `GLM-UCB`, and `ada-OFU-ECOLog-TS`.
For the first three algorithms, TS is automatically triggered for unit ball arm-set.

## Single experiment 
Single experiments (one algorithm for one environment) can be run thanks to `scripts/run_example.py`. The script instantiate the algorithm and environment indicated in the file `scripts/configs/example_config.py` and plots the regret.

## Reproducing the experiments
The results in the paper can be obtained thanks to `scripts/run_all.py`. This script runs experiments for any config file in `scripts/configs/generated_configs/` and stores the result in `scripts/logs/`.


## Plot results
### Regret curves
You can use `scripts/plot_regret.py` to plot regret curves. This scripts plot regret curves for all logs in `scripts/logs/` that match the indicated dimension and parameter norm. 

```
usage: plot_regret.py [-h] [-d [D]] [-hz [HZ]] [-ast [AST]] [-pn [PN]]

Plot regret curves

optional arguments:
  -h, --help  show this help message and exit
  -d [D]      Dimension (default: 2)
  -hz [HZ]    Horizon length (default: 4000)
  -ast [AST]  Dimension (default: tv_discrete)
  -pn [PN]    Parameter norm (default: 9.0)
```

Example output (S=10):

<img src="./regret_fig.png" width="500" alt="">


### Confidence sets
You can use `scripts/plot_confidence.py` to plot confidence sets. This scripts plot confidence sets for all logs in `scripts/S=*`.

```
usage: plot_confidence.py [-h] [-ast [AST]] [-pn [PN]] [-Nconfidence [N]]

Plot confidence sets for all algorithms

optional arguments:
  -h, --help          show this help message and exit
  -ast [AST]          Dimension (default: tv_discrete)
  -pn [PN]            Parameter norm (default: 9.0)
  -Nconfidence [N]    Number of discretizations (per axis) for confidence set plot (default: 5000)
```

Example output (S=10):

<img src="./confidence_fig.png" width="500" alt="">


## Generating configs 
You can automatically generate config files thanks to `scripts/generate_configs.py`. 

```
usage: generate_configs.py [-h] [-dims DIMS [DIMS ...]] [-pn PN [PN ...]] [-algos ALGOS [ALGOS ...]] [-r [R]] [-hz [HZ]] [-ast [AST]] [-ass [ASS]] [-fl [FL]]

Automatically creates configs, stored in configs/generated_configs/

optional arguments:
  -h, --help            show this help message and exit
  -dims DIMS [DIMS ...]
                        Dimension (default: None)
  -pn PN [PN ...]       Parameter norm (||theta_star||) (default: None)
  -algos ALGOS [ALGOS ...]
                        Algorithms. Possibilities include GLM-UCB, LogUCB1, OFULog-r, OL2M, GLOC, adaECOLog, or OFULogPlus (default: None)
  -r [R]                # of independent runs (default: 20)
  -hz [HZ]              Horizon, normalized (later multiplied by sqrt(dim)) (default: 1000)
  -ast [AST]            Arm set type. Must be either fixed_discrete, tv_discrete or ball (default: fixed_discrete)
  -ass [ASS]            Arm set size, normalized (later multiplied by dim) (default: 10)
  -fl [FL]              Failure level, must be in (0,1) (default: 0.05)
  -Nconfidence [N]      Number of discretizations (per axis) for confidence set plot
```

For instance running `python generate_configs.py -dims 2 -pn 3 4 5 -algos GLM-UCB GLOC OL2M adaECOLog` generates configs in dimension 2 for `GLM-UCB`, `GLOC`, `OL2M` and `adaECOLog`, for environments (set as defaults) of ground-truth norm 3, 4 and 5.



