import sys
import numpy as np
import tensorflow as tf
from config import Config
from models import resolve_model
from rewards import resolve_reward
from env import Maze, resolve_env

class NES():

	def __init__(self):
		self.config = Config().config
		self.env = resolve_env(self.config['environment'])(
			[[3,0,0,0,0,0,0,0,1,1],
			 [1,0,0,0,1,0,1,0,1,1],
			 [1,1,1,0,1,0,1,0,1,1],
			 [1,1,1,0,0,0,0,0,1,1],
			 [1,1,0,1,1,0,0,1,1,1],
			 [1,1,0,1,1,0,0,0,1,1],
			 [1,0,0,0,0,0,1,0,1,1],
			 [1,1,1,0,1,0,1,0,1,1],
			 [1,1,1,0,0,0,4,0,1,1],
			 [1,1,1,1,1,1,0,0,1,1]])
		if not self.env.solution_exists():
			raise Exception("No solution exists for the given environment.")
		self.model = resolve_model(self.config['model'])
		self.reward = resolve_reward(self.config['reward'])
		self.master_params = self.model.init_master_params()
		self.total_weighted_noise = np.zeros(len(self.master_params))

	def run_simulation(self, sample_params, model):
		with tf.Session() as sess:
			reward = 0
			for t in range(self.config['n_timesteps_per_trajectory']):
				inputs = np.array([self.env.current[0], self.env.current[1], self.env.target[0], self.env.target[1], self.env.is_wall(self.env.current, 0), self.env.is_wall(self.env.current, 1), self.env.is_wall(self.env.current, 2), self.env.is_wall(self.env.current, 3)]).reshape((1, 8))
				action_dist = sess.run(model, self.model.feed_dict(inputs, sample_params))
				# print("INPUTS:", inputs)
				# print("ACTION DIST:", action_dist)
				direction = np.argmax(action_dist)
				self.env.move(direction)
				reward += self.reward(self.env.current, self.env.target)
				print("Action: {}, Current Position: {}, Target Position: {}".format(self.env.direction_name(direction), self.env.current, self.env.target))
			self.env.reset()
			return reward

	def update(self, noise_samples, rewards):
		alpha = self.config['learning_rate']
		n_individuals = self.config['n_individuals']
		sigma = self.config['noise_std_dev']
		# print(np.std(rewards))
		# if np.std(rewards) == 0.0:
		# 	print(rewards)
		# 	print("NUM: ", (rewards - np.mean(rewards)))
		# 	print("STD of Rewards = 0")
		# 	sys.exit(1)
		# normalized_rewards = np.random.normal(0, 1, len(rewards))
		normalized_rewards = (rewards - np.mean(rewards))
		# print(normalized_rewards)
		if np.std(rewards) != 0.0:
			normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
		self.master_params += (alpha / (n_individuals * sigma)) * np.dot(noise_samples.T, normalized_rewards)
		self.total_weighted_noise = np.zeros(len(self.master_params))

	def run(self):
		model = self.model.model()
		for p in range(self.config['n_populations']):
			print("Population: {}".format(p+1))
			noise_samples = np.random.randn(self.config['n_individuals'], len(self.master_params))
			rewards = np.zeros(self.config['n_individuals'])
			for i in range(self.config['n_individuals']):
				print("Individual: {}".format(i+1))
				sample_params = self.master_params + noise_samples[i]
				rewards[i] = self.run_simulation(sample_params, model)
				print("Individual {} Reward: {}\n".format(i+1, rewards[i]))
				print("Max master_params: {}".format(max(self.master_params)))
			self.update(noise_samples, rewards)