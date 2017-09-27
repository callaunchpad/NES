import numpy as np
import tensorflow as tf
from algorithm import NES

if __name__ == "__main__":
	np.random.seed(0)
	tf.set_random_seed(0)
    algorithm = NES()
    algorithm.run()