# Simplified Temporal Consistency Reinforcement Learning

This is the PyTorch implementation of Simplified Temporal Consistency Reinforcement Learning (TCRL). 
## How to run the code
### Install dependencies
```shell
conda env create -f environment.yaml
conda activate tcrl
```

### Run the code
```python
python main.py task=walker_walk
```
To log metircs with wandb
```python
python main.py task=walker_walk use_wandb=true
```
You can also save video, replay buffer, trained model and logging by setting `save_<video/buffer/model/logging>=true`. All tested tasks are listed in `cfgs/task`

## Acknowledgements
We thanks the [TD-MPC](https://arxiv.org/abs/2203.04955) and [DrQv2](https://arxiv.org/abs/2107.09645) authors for their high-quality source code.

