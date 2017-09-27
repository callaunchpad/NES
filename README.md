# NES
Simple Feed Forward Example of a Natural Evolution Strategy as described by the [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf) paper authored by OpenAI

This is a good example of a modular separation between environments, training, model building, etc. in order to allow for a plug-and-play style of machine learning.

Only `Config.yaml` should be changed between different training examples. Major code changes should come with a version number.

## Known Issues
	- Converges to sub-optimal policy of moving right until stuck.
	- Resolve gaussian standardization of rewards when std dev of rewards is 0 (Converged to policy).