import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import shutil
import yaml
from datetime import datetime


def check_file_type(value: str) -> str:
    assert value in ('mat_dannce', 'mat_qualisys', 'simple_csv', 'dlc_csv', 'npy', 'df3d_pkl', 'sleap_h5')
    return value

OmegaConf.register_new_resolver("file_type", check_file_type)


@hydra.main(version_base=None, config_path="../conf", config_name="config_create_project")
def cli(_cfg: DictConfig) -> None:

    for key in ('working_directory', 'project_name', 'data_files', 'file_type'):
        val = _cfg[key]
        if val is None:
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-create-project dir=mydir project_name=test_project input_files=[x,y,'
                  f'z] file_type=simple_csv\n')
            sys.exit(1)


    main(
        project_name=_cfg.project_name,
        working_directory=_cfg.working_directory,
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


def main(project_name: str,
         working_directory: str,
         data_file_list: list,
         file_type: str):

    ## CHECK FILE VALIDITY
    for f in data_file_list:
        if not os.path.exists(f):
            print(f'\n❌ File {f} not found. Please check path.\n')
            sys.exit(1)

        if not check_extension(f, file_type):
            print(f'\n❌ File {f} does not have the correct extension. Should be of type {file_type}.\n')
            sys.exit(1)

    ## CREATE DISK PROJECT FOLDER
    project_path = os.path.join(working_directory, project_name)
    ext_project_path = 1
    final_project_path = str(project_path)
    while os.path.exists(final_project_path):
        final_project_path = project_path + f'_{ext_project_path}'
        ext_project_path += 1

    os.mkdir(final_project_path)
    os.mkdir(os.path.join(final_project_path, 'DISK_data'))
    os.mkdir(os.path.join(final_project_path, 'DISK_train'))
    os.mkdir(os.path.join(final_project_path, 'DISK_impute'))

    ### COPY EXAMPLE CONFIGS IN SUBDIRECTORY
    example_configs_folder = os.path.join(final_project_path, 'example_configs')
    os.mkdir(example_configs_folder)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    for config in ('config_prepare_data', 'config_train', 'config_impute'):
        shutil.copy(os.path.join(script_directory, '..', 'conf', f'{config}.yaml'),
                    os.path.join(example_configs_folder, f'{config}.yaml'))

    ### CREATE DISK_PROJECT_LOG
    default_config = {
        'working_directory': working_directory,
        'project_name': project_name,
        'data_files': list(data_file_list),
        'file_type': file_type,
        'skeleton': '',
        'creation_date': datetime.today().strftime('%Y-%m-%d'),
    }

    output_file = os.path.join(final_project_path, 'config_project.yaml')

    # Save the default configuration to a YAML file
    with open(output_file, 'w') as file:
        yaml.dump(default_config, file)

    if final_project_path != project_path:
        print(f'\n✅ A DISK project under the specified name already exists.'
              f'\n  Created DISK project {final_project_path}\n')
    else:
        print(f'\n✅ Created DISK project {final_project_path}\n')



if __name__ == '__main__':
    cli()
    """
    # hydra syntax:
    DISK-create-project dir=mydir project_name=test_project input_files=[x,y,z] file_type=csv
    """