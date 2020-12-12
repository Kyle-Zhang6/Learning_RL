## GYM - CARTPOLE

- #### RL_Functions:

  Functions to be used...

  __'./agents/'__: contains python classes of __DQN_AGENT__ and __VPG_AGENT__;

  __'./neuralnetwork'__: using pytorch to create nn models.

- #### results:

  Saving all the training process (rewards-episode figs) in this dir.

- #### Code_Files:

  __'./DQN.py'__: Implemented DQN;
  
  __'./DQN_MultiAgent.py'__: Multiple agents sharing the same experience pool;

  __'./VPG.py'__: Vanilla Policy Gradient, with or without baseline (more stable when with the baseline);

  __'./actor_critic_action_value.py'__: AC. Critic predicts the quality for each action;

  __'./A2C.py'__: Advantage Actor-Critic;