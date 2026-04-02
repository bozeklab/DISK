import os
import sys
import shutil
import yaml
from datetime import datetime
import argparse

def main(project_path: str):

    ### COPY EXAMPLE CONFIGS IN SUBDIRECTORY
    example_configs_folder = os.path.join(project_path, 'example_configs')
    os.makedirs(example_configs_folder, exist_ok=True)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    for config in ('config_prepare_data', 'config_train', 'config_evaluate', 'config_impute'):
        shutil.copy(os.path.join(script_directory, '..', 'conf', f'{config}.yaml'),
                    os.path.join(example_configs_folder, f'{config}.yaml'))

    print(f'\n✅ Restored default config for DISK project {project_path}\n')


def cli() -> None:
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--project_path', type=str, help='tile output width, in pixels', nargs='?')

    args = parser.parse_args()

    try:
        project_path = args.project_path
    except ValueError:
            print(f'\n❌ No value was passed to parameter {project_path}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-restore-config --project_path path/to/project')
            sys.exit(1)

    ## CHECK FILE VALIDITY
    if not os.path.exists(project_path):
        print(f'\n❌ Path {project_path} not found. Please check path.\n')
        sys.exit(1)

    main(
        project_path=project_path,
    )


if __name__ == '__main__':
    cli()
    """
    # hydra syntax:
    DISK-restore-config --project_path path/to/project
    """