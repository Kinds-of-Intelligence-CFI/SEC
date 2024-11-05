# SEC

Code repository for the research paper titled "Sequential memory improves sample and memory efficiency in Episodic Control".
Authors: Ismael T. Freire, Adrián F. Amil and Paul F.M.J. Verschure.

# Requirements

- animalai 1.1.1 
- animalai-train 1.1.1
- stable-baselines3 1.7.0
- tensorflow 1.14.0
- keras 2.2.5
- h5py 2.10
- grpcio 1.47.0
- torch 1.12.1
- gym 0.21
- numpy 1.21.6


# Simulation environment

To run the simulations, the creation of a conda environment is recommended. Proceed by following these instructions:

1. Select the model you want to run, and edit its parameters on the corresponding script: 

	- For SEC: run_agent_sec.py
	- For DQN: run_agent_dqn.py
	- For MFEC: run_agent_mfec.py
	- For ERLAM: run_agent_sec.py

2. Once you have selected the number of experiments and model parameters, run the script.

3. The data from the experiments will be stored in /data/simulations.

4. Use the notebook files in /notebooks for analyze the data and plot the figures.


# About the Animal-AI environment

For more details about the version used in this paper, check this forked version: https://github.com/IsmaelTito

For more information about the Animal-AI environment, visit: 
- Animal-AI v2: https://github.com/beyretb/AnimalAI-Olympics
- Animal-AI v3: https://github.com/Kinds-of-Intelligence-CFI/animal-ai

