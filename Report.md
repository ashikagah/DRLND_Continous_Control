# Continuous Control

<img src="balls.png" width="70%" align="top-left" alt="" title="Final Result" />

## Goal
To build a deep reinforcement learning agent that controls 20 robotic arms to maintain contact with the green balls. A reward of +0.1 is provided for each timestep when the agent's hand is in contact with the green balls. The environment is considered solved when the agents achive a mean score of +30.

## Learning Algorithm
To get started, there are a few high-level architecture decisions we need to make. First, we need to determine which types of algorithms are most suitable for the Reacher environment. Second, we need to determine how many "brains" we want controlling the actions of our agents.

### Deep Deterministic Policy Gradient (DDPG)
The [DDPG algorithm](https://arxiv.org/pdf/1509.02971.pdf) was chosen. The [DDPG template](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) provided in the ND repo was used to develop the algorithm.  

The Ornstein-Uhlenbeck process was used to add noise to the action values at each timestep [ref](https://arxiv.org/pdf/1509.02971.pdf). 

In total, there are five hyperparameters related to this noise process. The final parameters were as follows:
```
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process
```

### Learning Interval [(code)](https://github.com/ashikagah/DRLND_Continuous_Control/blob/master/ddpg_agent.py)
The learning timestep interval was set to 20 to avoid slow learning. The algorithm samples experiences from the buffer and learns 10 times.

```
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 10          # number of learning passes
```

### Gradient Clipping [(code)](https://github.com/ashikagah/DRLND_Continuous_Control/blob/master/ddpg_agent.py)
Gradient clipping was used to place an upper limit on the size of the parameter updates.

### Batch Normalization [(code)](https://github.com/ashikagah/DRLND_Continuous_Control/blob/master/model.py)
Batch normalization was used at the outputs of the first fully-connected layers of both the actor and critic models.

### Experience Replay [(code)](https://github.com/ashikagah/DRLND_Continuous_Control/blob/master/ddpg_agent.py)
Experience replay was also implemented.

## Plot of Rewards
<img src="plot.png" width="70%" align="top-left" alt="" title="Final Result" />

## Ideas for Future Work
Other algorithms such as [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/abs/1611.02247), [Advantage Actor-Critic (A2C)](https://openai.com/blog/baselines-acktr-a2c/), or [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) should be explored.






