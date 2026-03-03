import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import shutil
import yaml
from datetime import datetime

possible_file_type_values = ('mat_dannce', 'mat_qualisys', 'simple_csv', 'dlc_csv', 'dlc_h5', 'npy', 'df3d_pkl',
                          'sleap_h5')

def check_file_type(value: str) -> str:
    if value in possible_file_type_values:
        return True
    else:
        return False

OmegaConf.register_new_resolver("file_type", check_file_type)


@hydra.main(version_base=None, config_path="../conf", config_name="config_create_project")
def cli(_cfg: DictConfig) -> None:

    for key in ('project_path', 'data_files', 'file_type'):
        val = _cfg[key]
        if val is None:
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-create-project dir=mydir project_path=test_project input_files=[x,y,'
                  f'z] file_type=simple_csv\n')
            sys.exit(1)

    if _cfg['project_path'] is None or type(_cfg['project_path']) != str:
        print(f'\n❌ project_path should be string. Got {_cfg.project_path}.')
        sys.exit(1)
    else:
        if os.path.isabs(_cfg['project_path']):
            if not os.path.exists(os.path.dirname(_cfg['project_path'])):
                print(f'\n❌ project_path should be valid path. Got {_cfg.project_path}.')
                sys.exit(1)
    project_path = _cfg['project_path']
    ext_project_path = 1
    final_project_path = str(project_path)
    while os.path.exists(final_project_path):
        final_project_path = project_path + f'_{ext_project_path}'
        ext_project_path += 1

    main(
        project_path=final_project_path,
        data_file_list=_cfg.data_files,
        file_type=_cfg.file_type,
    )

def check_extension(file_path: str, file_type:str) -> bool:

    file_extension = os.path.splitext(file_path)[1]
    if file_type == 'mat_dannce':
        return file_extension == '.mat'
    elif file_type == 'mat_qualisys':
        return file_extension == '.mat'
    elif file_type == 'simple_csv':
        return file_extension == '.csv'
    elif file_type == 'npy':
        return file_extension == '.npy'
    elif file_type == 'df3d_pkl':
        return file_extension == '.pkl'
    elif file_type == 'dlc_csv':
        return file_extension == '.csv'
    elif file_type == 'dlc_h5':
        return file_extension == '.h5'
    elif file_type == 'sleap_h5':
        return file_extension == '.h5'
    else:
        return False


def main(project_path: str,
         data_file_list: list,
         file_type: str):

    ## CHECK FILE VALIDITY
    for f in data_file_list:
        if not os.path.exists(f):
            print(f'\n❌ File {f} not found. Please check path.\n')
            sys.exit(1)

        if not check_file_type(file_type):
            print(f'\n❌ File_type {file_type} is not correct. Should be one '
                  f'of {possible_file_type_values}.\n')
            sys.exit(1)

        if not check_extension(f, file_type):
            print(f'\n❌ File {f} does not have the correct extension. Should be of type {file_type}.\n')
            sys.exit(1)

    ## CREATE DISK PROJECT FOLDER
    os.mkdir(project_path)
    os.mkdir(os.path.join(project_path, 'DISK_data'))
    os.mkdir(os.path.join(project_path, 'DISK_train'))
    os.mkdir(os.path.join(project_path, 'DISK_impute'))

    ### COPY EXAMPLE CONFIGS IN SUBDIRECTORY
    example_configs_folder = os.path.join(project_path, 'example_configs')
    os.mkdir(example_configs_folder)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    for config in ('config_prepare_data', 'config_train', 'config_impute'):
        shutil.copy(os.path.join(script_directory, '..', 'conf', f'{config}.yaml'),
                    os.path.join(example_configs_folder, f'{config}.yaml'))

    ### CREATE DISK_PROJECT_LOG
    default_config = {
        'project_path': project_path,
        'data_files': list(data_file_list),
        'file_type': file_type,
        'creation_date': datetime.today().strftime('%Y-%m-%d'),
    }

    output_file = os.path.join(project_path, 'config_project.yaml')

    # Save the default configuration to a YAML file
    with open(output_file, 'w') as file:
        yaml.dump(default_config, file)

    if project_path != project_path:
        print(f'\n✅ A DISK project under the specified name already exists.'
              f'\n  Created DISK project {project_path}\n')
    else:
        print(f'\n✅ Created DISK project {project_path}\n')



if __name__ == '__main__':
    cli()
    """
    # hydra syntax:
    DISK-create-project project_path=test_project input_files=[x,y,z] file_type=csv
    # careful no space between input_files inside the brackets
    """