import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
from shutil import copyfile, rmtree
from datetime import datetime
from algorithm import NES

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Flags to toggle options")
	parser.add_argument('-d', action='store_true', default=False, help='Debug flag, Delete log after training is finished.')
	args = parser.parse_args()

	np.random.seed(0)
	tf.set_random_seed(0)
	timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
	try:
		os.makedirs("ext/" + timestamp)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
	training_directory = "ext/" + timestamp + "/"
	log_file = training_directory + timestamp + '.log'
	logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(message)s')
	copyfile("Config.yaml", training_directory + timestamp + ".yaml")

	try:
		algorithm = NES(training_directory)
		print("Running NES Algorithm...")
		print("Check {} for progress".format(log_file))
		algorithm.run()
	except KeyboardInterrupt:
		if args.d:
			print("\nDeleted: {}".format(training_directory))
			rmtree(training_directory)
		sys.exit(1)
	
	if args.d:
		print("\nnDeleted: {}".format(training_directory))
		rmtree(training_directory)
