# Two-Month Intensive Reinforcement Learning Syllabus for Neuromechanics of Movement

---

| Disclaimer: this syllabus was derived after much proooooompting with GPT-4o. 

## Overview

This intensive 8-week program is designed to rapidly build your expertise in Reinforcement Learning (RL) with a focus on applications in the neuromechanics of movement. By the end of this course, you will have a deep understanding of RL algorithms, practical implementation skills, and experience applying RL to biomechanical systems relevant to Professor Seungmoon Song's research.

---

## Week 1: Foundations of Reinforcement Learning

**Objectives:**

- Grasp the core concepts of RL: Markov Decision Processes (MDPs), policies, value functions, and the Bellman equations.
- Implement basic RL algorithms in simple environments.

**Topics to Cover:**

- Introduction to Reinforcement Learning
- Elements of RL: Agent, Environment, Reward, State, Action
- Markov Decision Processes (MDPs)
- Bellman Equations
- Dynamic Programming methods: Policy Evaluation, Policy Iteration, Value Iteration

**Readings:**

- **Book:** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd Edition), Chapters 1-3.
- **Lecture Series:** David Silver's RL Course, Lectures 1-2.
  - [Lecture Videos](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOk8)
  - [Lecture Slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- **Blog Posts:**
  - ["A (Long) Peek into Reinforcement Learning" by Lilian Weng](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)

**Assignments:**

- **Theory Exercises:**
  - Solve problems at the end of Chapters 2 and 3 in Sutton & Barto.
  - Prove that policy iteration converges to the optimal policy.
- **Coding Projects:**
  - Implement a Gridworld environment from scratch.
  - Write a simple agent using a random policy.
  - Compute state-value functions using Dynamic Programming.
  - Implement Policy Iteration and Value Iteration algorithms.
  - Visualize the optimal policy and value functions.

**Tools and Resources:**

- Python, NumPy
- OpenAI Gym for baseline environments
- Matplotlib for visualization

---

## Week 2: Classical RL Algorithms

**Objectives:**

- Understand Temporal Difference (TD) learning, Monte Carlo methods, SARSA, and Q-learning.
- Learn about the exploration-exploitation trade-off and ε-greedy policies.

**Topics to Cover:**

- Monte Carlo Methods
- Temporal Difference Learning
- SARSA vs. Q-Learning
- Exploration Strategies: ε-greedy, Softmax
- On-policy vs. Off-policy Learning

**Readings:**

- **Book:** Sutton & Barto, Chapters 5-7.
- **Research Paper:** Watkins, C. J. C. H., & Dayan, P. (1992). *Q-Learning*.
- **Lecture Series:** David Silver's RL Course, Lectures 3-4.
- **Blog Posts:**
  - ["Understanding SARSA and Q-Learning" by Arthur Juliani](https://medium.com/emergent-future/simple-reinforcement-learning-q-learning-fcddc4b6fe56)

**Assignments:**

- **Theory Exercises:**
  - Derive the update rules for SARSA and Q-learning.
  - Explain the convergence properties of Q-learning.
- **Coding Projects:**
  - Implement SARSA and Q-learning on the CliffWalking environment.
  - Compare the performance of both algorithms.
  - Experiment with different exploration strategies.
  - Plot learning curves showing cumulative rewards over episodes.

**Tools and Resources:**

- OpenAI Gym's `CliffWalking-v0` environment
- Python, NumPy, Matplotlib

---

## Week 3: Function Approximation in RL

**Objectives:**

- Understand the need for function approximation in large state spaces.
- Learn linear and non-linear (neural network) function approximators.
- Understand the bias-variance trade-off in function approximation.

**Topics to Cover:**

- Limitations of Tabular Methods
- Function Approximation Basics
- Linear Function Approximation
- Non-linear Function Approximation with Neural Networks
- Stochastic Gradient Descent in RL
- Eligibility Traces and TD(λ)

**Readings:**

- **Book:** Sutton & Barto, Chapters 9-10.
- **Online Tutorial:** *Neural Networks and Deep Learning* by Michael Nielsen, Chapters 1-3.
  - [Online Book](http://neuralnetworksanddeeplearning.com/)
- **Lecture Series:** David Silver's RL Course, Lectures 5-6.
- **Research Paper:** Tsitsiklis, J. N., & Van Roy, B. (1997). *An Analysis of Temporal-Difference Learning with Function Approximation*.

**Assignments:**

- **Theory Exercises:**
  - Explain the bias-variance trade-off in the context of function approximation.
  - Derive the gradient of the mean squared value error with respect to the parameters.
- **Coding Projects:**
  - Implement a linear function approximator for the MountainCar environment.
  - Extend it to a neural network approximator using TensorFlow or PyTorch.
  - Compare performance between linear and non-linear approximators.
  - Experiment with different network architectures and activation functions.

**Tools and Resources:**

- TensorFlow or PyTorch
- OpenAI Gym's `MountainCar-v0` environment
- Scikit-learn for linear regression

---

## Week 4: Deep Reinforcement Learning

**Objectives:**

- Dive into Deep Q-Networks (DQN) and experience replay.
- Understand stability issues in training deep RL agents.
- Learn techniques to stabilize training.

**Topics to Cover:**

- Deep Q-Networks (DQN)
- Experience Replay
- Fixed Q-Targets
- Double DQN
- Dueling DQN
- Prioritized Experience Replay

**Readings:**

- **Research Papers:**
  - Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*.
  - Van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning*.
  - Wang, Z. et al. (2015). *Dueling Network Architectures for Deep Reinforcement Learning*.
- **Lecture Series:** David Silver's RL Course, Lecture 7.
- **Videos:**
  - ["Train Deep Q-Learning on Atari in PyTorch" by brthor](https://www.youtube.com/watch?v=tsy1mgB7hB0)

**Assignments:**

- **Theory Exercises:**
  - Analyze the role of target networks in DQN.
  - Explain why experience replay improves learning efficiency.
- **Coding Projects:**
  - Implement DQN on the CartPole environment.
  - Implement Double DQN and Dueling DQN enhancements.
  - Experiment with Prioritized Experience Replay.
  - Tune hyperparameters to improve stability and performance.

**Tools and Resources:**

- TensorFlow or PyTorch
- OpenAI Gym's `CartPole-v1` environment
- TensorBoard for visualization

---

## Week 5: Policy Gradient Methods

**Objectives:**

- Learn about policy-based methods, specifically Policy Gradients and the REINFORCE algorithm.
- Understand Actor-Critic architectures.
- Explore variance reduction techniques like baseline functions.

**Topics to Cover:**

- Limitations of Value-Based Methods
- Policy Gradient Theorem
- REINFORCE Algorithm
- Actor-Critic Methods
- Advantage Functions
- Generalized Advantage Estimation (GAE)

**Readings:**

- **Book:** Sutton & Barto, Chapter 13.
- **Research Papers:**
  - Williams, R. J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*.
  - Schulman, J. et al. (2015). *Trust Region Policy Optimization*.
  - Schulman, J. et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*.
- **Lecture Series:** David Silver's RL Course, Lecture 8.
- **Blog Posts:**
  - ["Policy Gradient Algorithms" by OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
  - ["An intro to Advantage Actor Critic methods: let’s play Sonic the Hedgehog!"](https://medium.com/free-code-camp/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d)

**Assignments:**

- **Theory Exercises:**
  - Derive the Policy Gradient Theorem.
  - Explain the purpose of the baseline in variance reduction.
- **Coding Projects:**
  - Implement the REINFORCE algorithm on the `LunarLander-v2` environment.
  - Extend it to an Actor-Critic model.
  - Implement Generalized Advantage Estimation (GAE).
  - Compare the performance of REINFORCE and Actor-Critic methods.

**Tools and Resources:**

- TensorFlow or PyTorch
- OpenAI Gym's `LunarLander-v2` environment

---

## Week 6: Advanced Deep RL Algorithms

**Objectives:**

- Explore advanced algorithms: Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and Twin Delayed DDPG (TD3).
- Understand their advantages in continuous control tasks.
- Learn about entropy regularization and stability in policy updates.

**Topics to Cover:**

- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
- Soft Actor-Critic (SAC)
- Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Importance Sampling
- Entropy Regularization

**Readings:**

- **Research Papers:**
  - Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*.
  - Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.
  - Fujimoto, S. et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*.
- **Blog Posts:**
  - ["Understanding PPO" by OpenAI](https://openai.com/blog/openai-baselines-ppo/)
  - ["Demystifying Soft Actor-Critic (SAC)" by Towards Data Science](https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665)
  - ["An Introduction to DDPG and TD3" by Arthur Juliani](https://awjuliani.medium.com/)
- **Lecture Series:**
  - Sergey Levine's Deep RL Course, Lectures on Advanced Policy Gradients
    - [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)

**Assignments:**

- **Theory Exercises:**
  - Compare and contrast PPO and TRPO.
  - Explain how SAC incorporates entropy into the objective function.
- **Coding Projects:**
  - Implement PPO on the `BipedalWalker-v3` environment.
  - Implement SAC on a continuous control task (e.g., `Pendulum-v0`).
  - Analyze performance metrics like sample efficiency and convergence speed.
  - Experiment with hyperparameters like clipping in PPO and entropy coefficients in SAC.

**Tools and Resources:**

- TensorFlow or PyTorch
- OpenAI Gym's `BipedalWalker-v3` and `Pendulum-v0` environments
- Weights & Biases or TensorBoard for experiment tracking

---

## Week 7: RL in Robotics and Continuous Control

**Objectives:**

- Apply RL algorithms to robotic control tasks.
- Learn to use physics simulators like MuJoCo or PyBullet.
- Understand Sim-to-Real transfer challenges.

**Topics to Cover:**

- Continuous Action Spaces
- Physics Simulators (MuJoCo, PyBullet)
- Domain Randomization
- Sim-to-Real Transfer
- Model-Based vs. Model-Free RL in Robotics

**Readings:**

- **Tutorials:**
  - OpenAI's *Spinning Up in Deep RL*, focusing on continuous control.
    - [Spinning Up](https://spinningup.openai.com/en/latest/)
  - Introduction to MuJoCo and PyBullet.
    - [MuJoCo Documentation](http://www.mujoco.org/book/index.html)
    - [PyBullet Quickstart Guide](https://pybullet.org/wordpress/guide/)
- **Research Papers:**
  - Lillicrap, T. P. et al. (2015). *Continuous control with deep reinforcement learning*.
  - Tobin, J. et al. (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*.
- **Blog Posts:**
  - ["Sim-to-Real Transfer in Robotics" by OpenAI](https://openai.com/blog/ingredients-for-robotics-research/)

**Assignments:**

- **Theory Exercises:**
  - Discuss the challenges of Sim-to-Real transfer in robotics.
  - Explain how domain randomization helps in transfer learning.
- **Coding Projects:**
  - Implement SAC on the `Humanoid-v2` environment in MuJoCo.
  - Experiment with domain randomization techniques.
  - Use PyBullet to simulate a simple robotic arm and train it to perform a pick-and-place task.
  - Compare performance between MuJoCo and PyBullet environments.

**Tools and Resources:**

- MuJoCo physics engine (30-day trial or student license)
- PyBullet (open-source alternative)
- OpenAI Gym MuJoCo environments
- TensorFlow or PyTorch

---

## Week 8: RL Applications in Neuromechanics of Movement

**Objectives:**

- Integrate RL with neuromechanical models.
- Understand the application of RL in simulating and controlling biomechanical systems.
- Prepare for contributing to research in Professor Seungmoon Song's lab.

**Topics to Cover:**

- Introduction to Biomechanics and Neuromechanics
- Musculoskeletal Modeling with OpenSim
- RL in Biomechanical Simulations
- Optimal Control in Movement Science
- Applications of RL in Prosthetics and Exoskeletons

**Readings:**

- **Research Papers:**
  - Review key publications from Professor Seungmoon Song's lab.
    - *Check the lab's website or Google Scholar for the most recent papers.*
  - Lee, D. D. et al. (2019). *A neuromechanical model of locomotion suggests a minimal set of control priorities accounts for the different strategies used to walk at slow and fast speeds*.
- **Tutorials and Documentation:**
  - OpenSim User Guide and Tutorials
    - [OpenSim Documentation](https://simtk-confluence.stanford.edu/display/OpenSim/OpenSim+Documentation)
  - "Using Reinforcement Learning in OpenSim Environments" by Stanford Neuromuscular Biomechanics Laboratory
    - [OpenSim RL Environments](https://github.com/stanfordnmbl/osim-rl)

**Assignments:**

- **Theory Exercises:**
  - Propose how RL can be used to optimize movement strategies in a musculoskeletal model.
  - Discuss the limitations and challenges of modeling human neuromechanics.
- **Capstone Project:**
  - Use OpenSim to build a simple musculoskeletal model (e.g., a single leg with muscles).
  - Apply an RL algorithm (e.g., PPO or SAC) to train the model to perform a movement task (e.g., walking, jumping).
  - Analyze the learned policy in terms of biomechanical efficiency and neural control strategies.
  - Prepare a report summarizing your methodology, results, and insights.

**Tools and Resources:**

- OpenSim software
- TensorFlow or PyTorch
- Access to computational resources (GPU recommended)
- Visualization tools (e.g., OpenSim's visualizer, Matplotlib)

---

## Additional Recommendations

**Supplementary Readings:**

- **Books:**
  - Levine, S. (2018). *Deep Reinforcement Learning*. Course notes from UC Berkeley.
    - [Course Notes](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/)
  - Peters, J., & Schaal, S. (2008). *Reinforcement Learning of Motor Skills with Policy Gradients*.
- **Surveys and Reviews:**
  - Kober, J., Bagnell, J. A., & Peters, J. (2013). *Reinforcement Learning in Robotics: A Survey*.
  - Deisenroth, M. P., Neumann, G., & Peters, J. (2013). *A Survey on Policy Search for Robotics*.

**Online Courses:**

- **Coursera:**
  - *Deep Learning Specialization* by Andrew Ng.
  - *Fundamentals of Neuroscience* by Harvard University.
- **Udacity:**
  - *Deep Reinforcement Learning Nanodegree*.
- **edX:**
  - *Computational Neuroscience* by University of Washington.

**Community Engagement:**

- Join forums like Reddit's [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/).
- Participate in discussions on OpenAI's [Spinning Up community](https://github.com/openai/spinningup).
- Attend virtual seminars or webinars on RL and neuromechanics.

**Final Notes:**

- **Consistency is Key:** Allocate at least 6-8 hours daily to studying and coding.
- **Hands-On Practice:** Focus on implementing algorithms from scratch to solidify understanding.
- **Research Integration:** Continuously relate your learning to applications in neuromechanics to maintain relevance.
- **Documentation:** Keep a detailed journal of your learnings, challenges, and solutions.
- **Peer Review:** Discuss your projects with peers or mentors to gain feedback.

---

## Outcome

By following this intensive syllabus, you will acquire a strong theoretical foundation and practical skills in reinforcement learning, tailored to the neuromechanics of movement. You will be well-prepared to contribute to Professor Seungmoon Song's lab and engage in cutting-edge research.

---