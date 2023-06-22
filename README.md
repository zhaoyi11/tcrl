# TCRL: Simplified Temporal Consistency Reinforcement Learning

This is the PyTorch implementation of Simplified Temporal Consistency Reinforcement Learning (TCRL).

## Method
TCRL shows that, a simple representation learning approach relying only on a latent dynamics model trained
by latent temporal consistency is sufficient for high-performance RL. This applies when using
pure planning with a dynamics model conditioned on the representation, but, also when utilizing
the representation as policy and value function features in model-free RL. In experiments, our
approach learns an accurate dynamics model to solve challenging high-dimensional locomotion tasks with online planners while being 4.1× faster
to train compared to ensemble-based methods.
With model-free RL without planning, especially on high-dimensional tasks, such as the DeepMind Control Suite Humanoid and Dog tasks, our approach outperforms model-free methods by a
large margin and matches model-based methods’ sample efficiency while training 2.4× faster.

![image](./media/TCRL.png)

Model architecture for computing our temporal
consistency reinforcement learning (TCRL) latent state $\hat{z}_t$. 
At each time step t the model encodes observation ot into a latent
state $\hat{z}_t$ using a neural network. The latent dynamics model $\textbf{d}_{\phi}$ predicts the next latent state $\hat{z}_{t+1}$ using $\hat{z}_t$ and action $a_t$. 
A standard momentum encoder $\textbf{e}_{\theta^-}$ prevents collapse of the latent state representation. 
We use the cosine loss between the latent state and the momentum encoded latent state, denoted by a dashed line, for training. 
The reward function, omitted for clarity, is trained with the standard MSE loss. 
This simple model works surprisingly well providing a trained dynamics model for planning experiments,
and, in model-free RL experiments, the trained latent states $\hat{z}_t$ are used as inputs to policy and value functions. 

![video](./media/cartpole_swingup.gif)
![video](./media/pendulum_swingup.gif)
![video](./media/acrobot_swingup.gif)
![video](./media/cup_catch.gif)
![video](./media/reacher_hard.gif)
![video](./media/cheetah_run.gif)

![video](./media/fish_swim.gif)
![video](./media/hopper_stand.gif)
![video](./media/quadruped_walk.gif)
![video](./media/walker_run.gif)
![video](./media/humanoid_walk.gif)
![video](./media/dog_run.gif)


## Instructions
### Install dependencies
```shell
conda env create -f environment.yaml
conda activate tcrl
```

### Train the agent
```python
python main.py task=walker_walk
```
To log metircs with wandb
```python
python main.py task=walker_walk use_wandb=true
```
You can also save video, replay buffer, trained model and logging by setting `save_<video/buffer/model/logging>=true`. All tested tasks are listed in `cfgs/task`

Results are saved in `results/tcrl.csv`. The `results/plot.ipynb` file can be used to plot resutls.
## Citation
If you use this repo in your research, please consider citing our original paper:

```
@article{zhao2023simplified,
  title={Simplified Temporal Consistency Reinforcement Learning},
  author={Zhao, Yi and Zhao, Wenshuai and Boney, Rinu and Kannala, Juho and Pajarinen, Joni},
  journal={arXiv preprint arXiv:2306.09466},
  year={2023}
}
```

## Acknowledgements
We thanks the [TD-MPC](https://arxiv.org/abs/2203.04955), [SAC-PyTorch](https://github.com/denisyarats/pytorch_sac) and [DrQv2](https://arxiv.org/abs/2107.09645) authors for their high-quality source code.

