def resolve_env(name):
	envs = {
		"Maze": Maze
	}
	return envs[name]

"""
[[3,0,0,0,0,0,0,0,1,1],
[1,0,0,0,1,0,1,0,1,1],
[1,1,1,0,1,0,1,0,1,1],
[1,1,1,0,0,0,0,0,1,1],
[1,1,0,1,1,0,0,1,1,1],
[1,1,0,1,1,0,0,0,1,1],
[1,0,0,0,0,0,1,0,1,1],
[1,1,1,0,1,0,1,0,1,1],
[1,1,1,0,0,0,4,0,1,1],
[1,1,1,1,1,1,0,0,1,1]]
"""

class Maze():

	def __init__(self, matrix):
		self.walls = []
		self.empty = []
		self.start = (0,0)
		self.target = (0,0)
		self.boundaries = (len(matrix),len(matrix[0]))
		self.map = matrix
		for i in range(len(matrix)):
			for j in range(len(matrix[0])):
				if matrix[i][j] == 3:
					self.start = (i, j)
				if matrix[i][j] == 4:
					self.target = (i, j)
				if matrix[i][j] == 1:
					self.walls.append((i, j))
				else:
					self.empty.append((i, j))
		self.current = self.start

	def solution_exists(self):
		queue = [self.start]
		visited = []
		while queue:
			x, y = queue.pop(0)
			visited.append((x, y))
			print("Checked:", (x, y))
			if self.map_location(x, y) == 4:
				return True
			for i in range(4):
				next = self.next((x, y), i)
				if not self.is_wall((x, y), i) and next not in visited:
					queue.append(next)
		return False

	def reset(self):
		self.current = self.start

	def direction_name(self, n):
		if n == 0: return "^"
		if n == 1: return "v"
		if n == 2: return "<"
		if n == 3: return ">"

	def map_location(self, x,y):
		return self.map[y][x]

	def next(self, current_position, direction):
		if direction == 0:
			y_next = current_position[1] - 1
			if y_next < 0 or self.is_wall(current_position, 0):
				y_next = current_position[1]
			return (current_position[0], y_next)
		if direction == 1:
			y_next = current_position[1] + 1
			if y_next >= self.boundaries[1] or self.is_wall(current_position, 1):
				y_next = current_position[1]
			return (current_position[0], y_next)
		if direction == 2:
			x_next = current_position[0] - 1
			if x_next < 0 or self.is_wall(current_position, 2):
				x_next = current_position[0]
			return (x_next, current_position[1])
		if direction == 3:
			x_next = current_position[0] + 1
			if x_next >= self.boundaries[0] or self.is_wall(current_position, 3):
				x_next = current_position[0]
			return (x_next, current_position[1])

	def move(self, direction):
		self.current = self.next(self.current, direction)

	def is_wall(self, current_position, direction):
		if direction == 0:
			y_next = current_position[1] - 1
			if y_next < 0 or self.map_location(current_position[0], y_next) == 1:
				return 1
			return 0
		if direction == 1:
			y_next = current_position[1] + 1
			if y_next >= self.boundaries[1] or self.map_location(current_position[0], y_next) == 1:
				return 1
			return 0
		if direction == 2:
			x_next = current_position[0] - 1
			if x_next < 0 or self.map_location(x_next, current_position[1]) == 1:
				return 1
			return 0
		if direction == 3:
			x_next = current_position[0] + 1
			if x_next >= self.boundaries[0] or self.map_location(x_next, current_position[1]) == 1:
				return 1
			return 0
