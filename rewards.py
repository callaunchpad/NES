import math

def resolve_reward(name):
	rewards = {
		"manhattan_distance": manhattan_distance
	}
	return rewards[name]

def manhattan_distance(current, target):
	"""
	Manhattan distance from current position to target
	Args:
	    current (tuple): x, y coordinates of the current position
	    target (tuple): x, y coordinates of the target
	Returns:
		(float): Manhattan distance from current position to target
	"""
	return -(abs(target[0] - current[0]) + abs(target[1] - current[1]))