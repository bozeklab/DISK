class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



import torch

if torch.cuda.is_available():
    print(f'GPU available and accessible by pytorch')
else:
    print(f'{bcolors.WARNING}GPU not accessible by pytorch')

try:
    import DISK
    import hydra
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    import os
except Exception as e:
        print(f'{bcolors.WARNING}Problem with imports')