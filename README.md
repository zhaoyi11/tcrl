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
All tested tasks are listed in `cfgs/task`

Please check the `branch:use_dynamics` for experiments that use the latent dynamic model for planning or imporving the policy training.
