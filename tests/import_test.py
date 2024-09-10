import torch

if torch.cuda.is_available():
    print('GPU available and accessible by pytorch')
else:
    print('GPU not accessible by pytorch')

try:
    import DISK
    import hydra
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    import os
except Exception as e:
        print('Problem with imports')