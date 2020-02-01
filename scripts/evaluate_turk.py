#!/usr/bin/env python3

"""
Evaluate a trained model or bot
"""

import argparse
import csv
import gym
import time
import datetime

import babyai.utils as utils
from babyai.evaluate import evaluate_demo_agent, batch_evaluate, evaluate, evaluate_fixed_seeds
# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="name of the demos file (REQUIRED or --demos-origin or --model REQUIRED)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes of evaluation (default: 1000)")
parser.add_argument("--seed", type=int, default=int(1e9),
                    help="random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--contiguous-episodes", action="store_true", default=False,
                    help="Make sure episodes on which evaluation is done are contiguous")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")

parser.add_argument("--human", action="store_true", default=False,
                    help="use human-generated instructions rather than synthetic ones")
parser.add_argument("--turk_file", default=None,
                    help="file to read experiment specs from")
parser.add_argument("--proj", default=None,
                    help="style of projection to use")
parser.add_argument("--proj_file", default=None,
                    help="file to read projectable sentences from")


def main(args, seed, episodes):
    # Set seed for all randomness sources
    utils.seed(seed)

    # Define agent

    env = gym.make(args.env)
    env.seed(seed)
    agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)
    if args.model is None and args.episodes > len(agent.demos):
        # Set the number of episodes to be the number of demos
        episodes = len(agent.demos)

    if args.proj is not None:
        assert args.proj_file is not None

    if args.proj_file is not None:
        with open(args.proj_file, newline="") as reader:
            proj_sentences = reader.readlines()
    else:
        proj_sentences = None

    seeds = []
    orig_missions = []
    missions = []
    with open(args.turk_file, newline="") as reader:
        csv_reader = csv.reader(reader)
        header = next(csv_reader)
        i_seed = header.index("Input.seed")
        i_orig_dir = header.index("Input.cmd")
        i_mission = header.index("Answer.command")
        for row in csv_reader:
            seeds.append(int(row[i_seed]))
            orig_missions.append(row[i_orig_dir])
            missions.append(row[i_mission])

    if not args.human:
        logs = evaluate_fixed_seeds(agent, env, episodes, seeds, orig_missions)
    else:
        logs = evaluate_fixed_seeds(agent, env, episodes, seeds, orig_missions, missions, args.proj, proj_sentences)

    return logs


if __name__ == "__main__":
    args = parser.parse_args()
    assert_text = "ONE of --model or --demos-origin or --demos must be specified."
    assert int(args.model is None) + int(args.demos_origin is None) + int(args.demos is None) == 2, assert_text

    start_time = time.time()
    logs = main(args, args.seed, args.episodes)
    end_time = time.time()

    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    ellapsed_time = int(end_time - start_time)
    duration = datetime.timedelta(seconds=ellapsed_time)

    if args.model is not None:
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])

    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    if args.model is not None:
        print("F {} | FPS {:.0f} | D {} | R:xsmM {:.3f} {:.3f} {:.3f} {:.3f} | S {:.3f} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration,
                      *return_per_episode.values(),
                      success_per_episode['mean'],
                      *num_frames_per_episode.values()))
    else:
        print("F {} | FPS {:.0f} | D {} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration, *num_frames_per_episode.values()))

    indexes = sorted(range(len(logs["num_frames_per_episode"])), key=lambda k: - logs["num_frames_per_episode"][k])

    n = args.worst_episodes_to_show
    if n > 0:
        print("{} worst episodes:".format(n))
        for i in indexes[:n]:
            if 'seed_per_episode' in logs:
                print(logs['seed_per_episode'][i])
            if args.model is not None:
                print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
            else:
                print("- episode {}: F={}".format(i, logs["num_frames_per_episode"][i]))
