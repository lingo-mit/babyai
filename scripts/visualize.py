#!/usr/bin/env python3

import gym

import babyai.utils as utils
import imageio

#MODEL = "BabyAI-Synth-v0_IL_expert_filmcnn_gru_seed1_20-01-08-21-13-27"
DEMOS = "BabyAI-SynthLoc-v0_valid"
ENV = "BabyAI-SynthLoc-v0"
SEED = 0

utils.seed(SEED)

env = gym.make(ENV)

global obs
obs = env.reset()

agent = utils.load_agent(env, None, DEMOS, None, False, ENV)

print(env.render().shape)
imageio.imsave("img.png", env.render())
