This repository contains Bayesian RL environments in OpenAI Gym API and BRL algorithms in OpenAI Baselines API.

It forks [OpenAI Baselines' tf2 branch](https://github.com/openai/baselines.git) for supporting Tensorflow2.2 with Python 3.8.

## Installation

1. Install Tensorflow 2.2.0

   ```bash
   python -m pip install virtualenv # Install virtualenv. You can use conda instead.
   virtualenv /path_to_venv/ -p python3.8
   . /path_to_venv/bin/activate
   python -m pip install tensorflow==2.2.0
   ```

2. Clone and install this repository

   ```bash
   git clone https://github.com/personalrobotics/bayesian_rl.git
   cd bayesian_rl
   git submodule init
   git submodule update
   python -m pip install -r requirements.txt
   ```

3. Test your installation. You should be able to run commands in OpenAI Baslines, e.g.

```bash
python -m baselines.run --alg=ppo2 --env=bayes-ContinuousCartPole-v0 --num_timesteps=1e6
```

Load and test one of the environments. The API is the same as OpenAI Gym:
```python
>>> from brl_gym import envs
>>> env = envs.Tiger()
>>> obs = env.reset()
>>> obs, reward, done, info = env.step(env.action_space.sample())
```
While the API is the same as a typical OpenAI Gym, the environments in Bayesian RL have latent variables which _can be estimated_ by a Bayes filter. We also provide Bayesian environments under `wrapper_envs` which wraps these environments and corresponding Bayes filters together to output `(observation, belief)`. The wrapper envs are the ones used in BRL algorithms. For every environment in `brl_gym.envs` we have its corresponding wrapper env in `brl_gym.wrapper_envs`:

```python
>>> from brl_gym import wrapper_envs
>>> bayes_env = wrapper_envs.BayesTiger()
>>> obs = bayes_env.reset()
```
This produces a longer observation vector than its corresponding environment, as it appends Bayes filter output (belief) to the original observation. 

The list of registered names can be found in `brl_gym/__init__.py`.


## Training with BPO
To run [Bayesian Policy Optimization](https://arxiv.org/abs/1810.01014) algorithm, we need to provide two additional parameters, 1) `--network=brl_mlp` to use the BPO network as introduced in the paper, and 2) the size of observation dimension. The latter information is used by `brl_mlp` to split up the gym's per-step output into observation and belief.

```bash
python -m baselines.run --alg=ppo2 --env=bayes-ContinuousCartPole-v0 --num_timesteps=2e7 --save_path=~/models/bpo-cartpole --network=brl_mlp --obs_dim=4 --num_env 2
python -m baselines.run --alg=ppo2 --env=bayes-ContinuousCartPole-v0 --num_timesteps=0 --load_path=~/models/bpo-cartpole --network=brl_mlp  --obs_dim=4 --play
```


For example, try the following:
```bash
source /path/to/venv/bin/activate

# BPO-
OPENAI_LOGDIR=~/models/cartpole/bpo_minus OPENAI_LOG_FORMAT=tensorboard python -m baselines.run --alg=ppo2 --env=bayes-ContinuousCartPole-v0 --num_timesteps=1e6 --save_path=~/models/bayes-cartpole-ppo --num_env 20 --save_interval 3
```
This trains a belief-aware agent with PPO algorithm. The checkpoints are saved in `OPENAI_LOGDIR`  and the checkpoints can be visualized by tensorboard:

```bash
cd ~/models/cartpole/bpo_minus/tb
tensorboard --logdir=.
```

To load the latest checkpoint,
```bash
python -m baselines.run --alg=ppo2 --env=bayes-ContinuousCartPole-v0 --num_timesteps=0 --load_path=~/models/cartpole/bpo_minus/checkpoints --play
```

## Bayes-Gym environments
See [brl_gym](brl_gym/brl_gym/README.md) for more detail.

## BPO without an explicit Bayes Filter
*This feature is temporarily disabled*.

Without an explicit Bayes Filter, you can directly take observations and account for history by training on LSTM networks. LSTM would internally maintain a feature which encodes the history of observations.

```bash
OPENAI_LOGDIR=~/models/cartpole/ppo_lstm OPENAI_LOG_FORMAT=tensorboard python -m baselines.run --alg=ppo2 --env=ContinuousCartPole-v0 --num_timesteps=2e7 --save_path=~/models/cartpole/ppo_lstm_final --network=lstm --num_env 20 --nminibatches 20
```
Note that this may require more training than one with a Bayes Filter. The current LSTM policy requires `num_env > nminibatches`. 


Please email Gilwoo Lee (gilwoo@uw.edu) for any bugs or questions.
