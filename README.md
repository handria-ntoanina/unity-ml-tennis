# unity-ml-reacher
This repository contains an implementation of deep reinforcement learning based on:
* Multi Agent Deep Deterministic Policy Gradients
* and Multi Agent Proximal Policy Optimization
	
The environment to be solved is having two agents playing tennis. Each agent is conducting a racket to bounce a ball over a net.
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
This environment is similar to the [tennis of Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).<br/>
The action space is continuous [-1.0, +1.0] and consists of 2 values for horizontal and jumping moves. <br/>
`The environment is considered as solved if the average score of one gent is >= 0.5 for 100 consecutive episodes.`<br/>
![Video](https://img.youtube.com/vi/6o7d0N5qyFs/0.jpg)

A video of trained agents can be found here below <br/>
* [MADDPG](https://www.youtube.com/watch?v=6o7d0N5qyFs)
* [MAPPO](https://www.youtube.com/watch?v=2y7yCrYfTXA)
## Content of this repository
* __analysis.xlsx__: results of several experiments
* __report.pdf__: a document that describes the implementation of the MADDPG and MAPPO, along with ideas for future work
* __run_tensorboard.bat__: to run tensorboard an visualize the loss during training
* folder __agents__: contains the implementation of
	* a Multi Agent DDPG
	* a Proximal Policy Optimization
	* an Actor-Critic network model using tanh as activation
	* A Gaussian based Actor-Critic network model using tanh as activation
	* a ReplayBuffer
	* Noise Generator
	    * an ActionNoise that disturb the output of the actor network to promote exploration
	    * a ParameterNoise that disturb the weight of the actor network to promote exploration
	    * an Ornstein-Uhlenbeck noise generator
	    * a simple noise generator based on numpy random generator
* folder __started_to_converge__: weights of a network that started to converge but slowly
* folder __final_weights__:
	* __final_maddpg_local_2.pth__ weights of a local network trained with MADDPG that solved this environment.
	* __final_maddpg_target_2.pth__ weights of a target network trained with MADDPG that solved this environment.
	* __final_maddpg_local.pth__ weights of a local network trained with MADDPG that reached 0.5 during the training but is not stable during visual validation.
	* __final_maddpg_target.pth__ weights of a target network trained with MADDPG that reached 0.5 during the training but is not stable during visual validation.
	* __final_ppo.pth__ weights of the Gaussian Actor Critic Network that solved this environment with Multi Agent PPO
	* __final_maddpg.png__ chart of the 1st phase of training using MADDPG
	* __final_maddpg_2.png__ chart of the 2st phase of training using MADDPG
    * __final_ppo.png__ chart of the result of the training using MAPPO
* Jupyter Notebooks
	* __Multi Agent Deep Deterministic Policy Gradient.ipynb__: run this notebook to train the agents using MADDPG and to view its performance
	* __Multi Agent Proximal Policy Optimization.ipynb__: run this notebook to train the agents using MAPPO and to view its performance

## Requirements
To run the codes, follow the next steps:
* Create a new environment:
	* __Linux__ or __Mac__: 
	```bash
	conda create --name ddpg python=3.6
	source activate ddpg
	```
	* __Windows__: 
	```bash
	conda create --name ddpg python=3.6 
	activate ddpg
	```
* Perform a minimal install of OpenAI gym
	* If using __Windows__, 
		* download swig for windows and add it the PATH of windows
		* install Microsoft Visual C++ Build Tools
	* then run these commands
	```bash
	pip install gym
	pip install gym[classic_control]
	pip install gym[box2d]
	```
* Install Tensorflow and Tensorboard
    ```bash
    pip install tensorflow, tensorflow-gpu
    ``` 
    or 
    ```bash
    pip install tensorflow
    ``` 
* Install PyTorch
    ```bash
    pip install pytorch
    ```
* Install the dependencies under the folder python/
```bash
	cd python
	pip install .
```
* Install jupyter notebook
```bash
	pip install jupyter notebook
```
* Fix an issue of pytorch 0.4.1 to allow backpropagate the torch.distribution.normal function up to its standard deviation parameter
    * change the line 69 of Anaconda3\envs\drlnd\Lib\site-packages\torch\distributions\utils.py
```python
# old line
# tensor_idxs = [i for i in range(len(values)) if values[i].__class__.__name__ == 'Tensor']
# new line
tensor_idxs = [i for i in range(len(values)) if isinstance(values[i], torch.Tensor)]
``` 
* Create an IPython kernel for the `ddpg` environment
```bash
	pip install ipykernel
	python -m ipykernel install --user --name ddpg --display-name "ddpg"
```
* If cannot start any notebook, run the following command to reinstall nbconvert
```bash
	pip3 install --upgrade --user nbconvert
```
* Download the Unity Environment (thanks to Udacity) which matches your operating system
	* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
* Start jupyter notebook from the root of this python codes
```bash
jupyter notebook
```
* Once started, change the kernel through the menu `Kernel`>`Change kernel`>`ddpg`
* If necessary, inside the ipynb files, change the path to the unity environment appropriately

