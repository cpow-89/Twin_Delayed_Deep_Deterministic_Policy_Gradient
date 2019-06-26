import os
import torch
import pybullet_envs
import gym
from agent import TwinDelayedDDPG
import numpy as np
import random
import json
import session
import utils


def main():
    with open(os.path.join(".", "configs", "AntBulletEnv_v0.json"), "r") as read_file:
        config = json.load(read_file)

    env = gym.make(config["env_name"])
    env.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TwinDelayedDDPG(config, state_dim, action_dim, max_action)

    if config["run_training"]:
        session.train(agent, env, config)
        agent.save()
    else:
        if config["create_video"]:
            env = utils.set_up_monitoring(env, config)
        agent.load()
        session.evaluate(agent, env, config["eval_episodes"])
    env.close()


if __name__ == '__main__':
    main()