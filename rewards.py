import math

def resolve_reward(name):
	rewards = {
		"manhattan_distance": manhattan_distance
	}
	return rewards[name]

def manhattan_distance(current, target):
	return -(abs(target[0] - current[0]) + abs(target[1] - current[1]))