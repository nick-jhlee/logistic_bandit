Codes for the following two papers by [Junghyun Lee](https://nick-jhlee.github.io/), [Se-Young Yun](https://fbsqkd.github.io/), and [Kwang-Sung Jun](https://kwangsungjun.github.io):
- [_A Unified Confidence Sequence for Generalized Linear Models, with Applications to Bandits_](https://arxiv.org/abs/2407.13977) (NeurIPS 2024),
- [_Improved Regret Bounds of (Multinomial) Logistic Bandits via Regret-to-Confidence-Set Conversion_](https://arxiv.org/abs/2310.18554) (AISTATS 2024).

This is forked from https://github.com/criteo-research/logistic_bandit.

If you plan to use this repository or cite our paper, please use the following bibtex format:

```latex
@inproceedings{lee2024glm,
	title={{A Unified Confidence Sequence for Generalized Linear Models, with Applications to Bandits}},
	author={Lee, Junghyun and Yun, Se-Young and Jun, Kwang-Sung},
	booktitle = {Advances in Neural Information Processing Systems},
	publisher = {Curran Associates, Inc.},
	volume = {37},
	year = {2024},
	url = {https://openreview.net/forum?id=MDdOQayWTA},
}

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

This code implements the following bandit algorithms (oldest to newest):

Only applicable for linear bandits (NOT YET IMPLEMENTED):
- `OFUL` ([Abbasi-Yadkori et al. 2011](https://papers.nips.cc/paper/2011/file/7f1c3f338b8e2e9d6b5e8e4e7f3b4208-Paper.pdf))
- `CMM-UCB` ([Flynn et al. 2023](https://openreview.net/forum?id=TXoZiUZywf&noteId=hiOPt0S3AO)) : not yet implemented..

Only applicable for logistic bandits:
- `OL2M` ([Zhang et al. 2016](http://proceedings.mlr.press/v48/zhangb16.pdf))
- `LogUCB1` ([Faury et al. 2020](http://proceedings.mlr.press/v119/faury20a/faury20a.pdf))
- `OFULog-r` ([Abeille et al. 2021](http://proceedings.mlr.press/v130/abeille21a/abeille21a.pdf))
- `ada-OFU-ECOLog` ([Faury et al. 2022](https://proceedings.mlr.press/v151/faury22a.html))
- `ada-OFU-ECOLog-TS` ([Faury et al. 2022](https://proceedings.mlr.press/v151/faury22a.html))
- `OFULog+` ([Lee et al. 2024a](https://arxiv.org/abs/2310.18554))

Only applicable for bounded GLBs:
- `GLM-UCB` ([Filippi et al. 2010](https://papers.nips.cc/paper/2010/file/c2626d850c80ea07e7511bbae4c76f4b-Paper.pdf)),
- `RS-GLinCB` ([Sawarni et al. 2024](https://arxiv.org/abs/2404.06831))

Generally applicable for any GLBs:
- `GLOC` ([Jun et al. 2017](https://proceedings.neurips.cc/paper/2017/file/28dd2c7955ce926456240b2ff0100bde-Paper.pdf)),
- `EMK` ([Emmenegger et al. 2023](hhttps://openreview.net/forum?id=4anryczeED))
- `EVILL` ([Janz et al. 2024](https://proceedings.mlr.press/v238/janz24a.html)) -- randomized exploration
- `OFUGLB` & `OFUGLB-e` ([Lee et al. 2024b](https://arxiv.org/abs/2407.13977))


Experiments can be run for several Logistic Bandit (i.e., structured Bernoulli feedback) environments, such as static and time-varying finite arm-sets, or inifinite arm-sets (_e.g._ unit ball).
Note that the Thompson Sampling type algorithm (TS) is only available for `GLOC`, `OL2M`, `GLM-UCB`, and `ada-OFU-ECOLog-TS`.
For the first three algorithms, TS is automatically triggered for unit ball arm-set.

## Single experiment 
Single experiments (one algorithm for one environment) can be run via `scripts/run_example.py`. The script instantiate the algorithm and environment indicated in the file `scripts/configs/example_config.py` and plots the regret.

## Reproducing the experiments
The results in the paper can be obtained via `scripts/run_all.py`, with the pre-generated configs. This script runs experiments for any config file in `scripts/configs/generated_configs/` and stores the result in `scripts/logs/`.


## Plot results
### Regret curves
You can use `scripts/plot_regret.py` to plot the regret curve for a specific S. This scripts plot regret curves for all logs in `scripts/logs/` that match the indicated dimension and parameter norm. 

```
usage: plot_regret.py [-h] [-d [D]] [-hz [HZ]] [-pn [PN]] [-ast [AST]]

Plot regret curves

optional arguments:
  -h, --help  show this help message and exit
  -d [D]      Dimension (default: 2)
  -hz [HZ]    Horizon length (default: 2000)
  -pn [PN]    Parameter norm (default: 3.0)
  -ast [AST]  Dimension (default: tv_discrete)
```


### Confidence sets
You can use `scripts/plot_confidence.py` to plot confidence sets. This scripts plot confidence sets for all logs in `scripts/S=*`.

```
usage: plot_confidence.py [-h] [-d [D]] [-hz [HZ]] [-pn [PN]] [-ast [AST]] [-Nconfidence [N]] [-algos [ALGOS [ALGOS ...]]]

Plot confidence sets for all algorithms

optional arguments:
  -h, --help          show this help message and exit
  -d [D]      Dimension (default: 2)
  -hz [HZ]    Horizon length (default: 2000)
  -pn [PN]            Parameter norm (default: 3.0)
  -ast [AST]          Dimension (default: tv_discrete)
  -Nconfidence [N]    Number of discretizations (per axis) for confidence set plot (default: 2000)
  -algos [ALGOS [ALGOS ...]]    Algorithms with which we will plot the confidence set (default: ['OFUGLB', 'OFUGLB-e', 'EMK' ,'OFULogPlus'])
```

### Plotting everything together
You can use `scripts/plot_total.py` to plot everything (Figure 1 of [Lee et al. 2024b](https://arxiv.org/abs/2407.13977))

```
usage: plot_total.py [-h] [-d [D]] [-hz [HZ]] [-pn [PN]] [-ast [AST]] [-Nconfidence [N]]

Plot regret curves

optional arguments:
  -h, --help  show this help message and exit
  -d [D]      Dimension (default: 2)
  -hz [HZ]    Horizon length (default: 2000)
  -pn [PN]    Parameter norm (default: 3.0)
  -ast [AST]  Dimension (default: tv_discrete)
  -Nconfidence [N]    Number of discretizations (per axis) for confidence set plot (default: 2000)
```

Example output:

<img src="./main_fig.png" width="1000" alt="">


## Generating configs 
You can automatically generate config files thanks to `scripts/generate_configs.py`. 

```
usage: generate_configs.py [-h] [-dims DIMS [DIMS ...]] [-pn PN [PN ...]] [-algos ALGOS [ALGOS ...]] [-r [R]] [-hz [HZ]] [-ast [AST]] [-ass [ASS]] [-fl [FL]]

Automatically creates configs, stored in configs/generated_configs/

optional arguments:
  -h, --help            show this help message and exit
  -dims DIMS [DIMS ...]
                        Dimension (default: None)
  -pn PN [PN ...]       Parameter norm (||theta_star|| = S - 1) (default: None)
  -algos ALGOS [ALGOS ...]
                        List of algorithms. (default: None)
                        ('OL2M', 'LogUCB1', 'OFULog-r', 'adaECOLog', 'OFULogPlus', 'GLM-UCB', 'RS-GLinCB', 'GLOC', 'EMK', 'EVILL', 'OFUGLB-e', 'OFUGLB')
  -r [R]                # of independent runs (default: 10)
  -hz [HZ]              Horizon length (default: 2000)
  -ast [AST]            Arm set type. Must be either fixed_discrete, tv_discrete, ball, or movielens (default: tv_discrete)
  -ass [ASS]            Arm set size (default: 10)
  -fl [FL]              Failure level, must be in (0,1) (default: 0.05)
  -plotconfidence [N]   To plot the confidence set at the end or not (default: False)
  -Nconfidence [N]      Number of discretizations (per axis) for confidence set plot
  -tol [TOL]            Tolerance for optimization (default: 1e-5)
  -seed [SEED]          Random seed for experiments (default: 9817)
  -theta_flag [THETA_FLAG]  1 for theta being aligned with the largest eigenvector and 2 for the opposite (default: None)
  -arm_option [ARM_OPTION]  '\'normalize\' for normalizing (default: None)
```

For instance running `python generate_configs.py -dims 2 -pn 3 4 5 -algos OFUGLB adaECOLog` generates configs in dimension 2 for `OFUGLB` and `adaECOLog`, for environments (set as defaults) of ground-truth norm 3, 4 and 5.



