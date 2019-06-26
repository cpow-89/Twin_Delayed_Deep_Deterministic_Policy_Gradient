import numpy as np


def evaluate(agent, env, eval_episodes=10):
    avg_reward = 0.
    for _ in range(int(eval_episodes)):
        transition = env.reset()
        done = False
        while not done:
            action = agent.act(np.array(transition))
            transition, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def _print_average_reward(total_timesteps, episode_num, eval_episodes, episode_reward):
    print("Total Timesteps: {} - "
          "Episode Num: {} - "
          "Average Reward over last {} - "
          "episodes: {}".format(total_timesteps, episode_num, eval_episodes, episode_reward))


def _add_exploration_noise(action, env, noise=None):
    if noise:
        return (action + np.random.normal(0, noise, size=env.action_space.shape[0])).clip(env.action_space.low,
                                                                                          env.action_space.high)


def train(agent, env, config):
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    best_reward = 0
    state = env.reset()
    episode_timesteps = 0

    while total_timesteps < config["n_timesteps"]:
        if total_timesteps < config["n_exploration_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.act(np.array(state))
            action = _add_exploration_noise(action, env, config["exploration_noise"])

        next_state, reward, done, _ = env.step(action)
        # Note: The check for _max_episode_steps prevents the agent from getting stuck
        # when he learned to stand without falling
        agent.add_transition_to_memory((state, next_state, action, reward,
                                        0.0 if episode_timesteps + 1 == env._max_episode_steps else float(done)))

        state = next_state
        total_timesteps += 1
        timesteps_since_eval += 1
        episode_timesteps += 1

        if total_timesteps > config["batch_size"]:
            agent.learn(total_timesteps)

        if done:
            if timesteps_since_eval >= config["evaluations_freq"]:
                timesteps_since_eval %= config["evaluations_freq"]
                average_reward = evaluate(agent, env, config["eval_episodes"])
                _print_average_reward(total_timesteps, episode_num, config["eval_episodes"], average_reward)

                if average_reward > best_reward:
                    print("Model saved")
                    agent.save()
                    best_reward = average_reward

            # reset
            state = env.reset()
            episode_timesteps = 0
            episode_num += 1
