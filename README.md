
A reinforcement learning problem of a simple model of a 2D manipulator that must reach a goal point with the end effector by sending
speed commands to joints (setting discrete angle translations).

The environment is based on the gym library (specifically, gym.GoalEnv). Learning and monitoring is performed using PyTorch, TensorFlow and Stable-Baselines3.
Visualization is done using Matplotlib.

The problem can be configured in several ways:
- fixed goal, fixed starting pose
- fixed goal, random starting pose
- random goal, fixed starting pose
- random goal, random starting pose

The training is configured for DQN with the possibility of using several environments in parallel and for DQN with HER on one environment. Note that
DQN-HER is useful mainly for the random goal problem, while fixed goal can be trained faster by using DQN with several environments.

# Installation

Install Python3 dependencies using the command:

```
pip3 install -r requirements.txt 
```

# Running

To see how the trained model performs or to train it use the `manipulator_2d.py` script:
- configure the problem on which to run it by changing the parameters in `make_env` function
- leave the required run function in the `main` function uncommented (for example, `train_dqn()` or `test_dqn_model()`)
- check the path to the trained model and its name (they are expected to be in the `models/2d_manipulator_model_GoalEnv/` subdirectory)
- run the `manipulator_2d.py` script from the `scripts` directory

```
cd scripts
python3 manipulator_2d.py
```

To monitor the learning process with various stats, use `tensorboard`:

```
tensorboard --logdir logs/logs_2d_manipulator_GoalEnv/
```

The trained models are saved to the `models/2d_manipulator_model_GoalEnv/` directory. During learning, the intermediate models are also saved to the
corresponding `checkpoints` directory.
