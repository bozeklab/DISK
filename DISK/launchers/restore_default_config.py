import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import shutil
import yaml
from datetime import datetime

def main(project_path: str):

    ### COPY EXAMPLE CONFIGS IN SUBDIRECTORY
    example_configs_folder = os.path.join(project_path, 'example_configs')
    os.makedirs(example_configs_folder, exist_ok=True)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    for config in ('config_prepare_data', 'config_train', 'config_test', 'config_impute'):
        shutil.copy(os.path.join(script_directory, '..', 'conf', f'{config}.yaml'),
                    os.path.join(example_configs_folder, f'{config}.yaml'))

    print(f'\n✅ Restored default config for DISK project {project_path}\n')


@hydra.main(version_base=None, config_path="../conf", config_name="config_restore_default_config")
def cli(_cfg: DictConfig) -> None:

    for key in ('project_path', ):
        val = _cfg[key]
        if val is None:
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-restore-config project_path=...\n')
            sys.exit(1)

    ## CHECK FILE VALIDITY
    if not os.path.exists(_cfg.project_path):
        print(f'\n❌ Path {_cfg.project_path} not found. Please check path.\n')
        sys.exit(1)

    main(
        project_path=_cfg.project_path,
    )


if __name__ == '__main__':
    cli()
    """
    # hydra syntax:
    DISK-restore-config project_path=...
    """