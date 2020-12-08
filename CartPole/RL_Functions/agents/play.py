def play(env, agent, render=False, train=False, extra_reward_func=None):
    observation = env.reset()
    episode_reward = 0
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if extra_reward_func is not None:
            reward += extra_reward_func(next_observation, reward, episode_reward, done)
        if train:
            agent.learn(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward