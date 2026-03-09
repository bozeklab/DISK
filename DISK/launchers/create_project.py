import os
import sys
import shutil
import yaml
from datetime import datetime
from glob import glob

from DISK.utils.config_decorator import config_reader, parse_command_line_args, StringList

possible_file_type_values = ('mat_dannce', 'mat_qualisys', 'simple_csv', 'dlc_csv', 'dlc_h5', 'npy', 'df3d_pkl',
                          'sleap_h5')

def check_file_type(value: str) -> str:
    if value in possible_file_type_values:
        return True
    else:
        return False

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


    ## CREATE DISK PROJECT FOLDER
    os.mkdir(project_path)
    os.mkdir(os.path.join(project_path, 'DISK_data'))
    os.mkdir(os.path.join(project_path, 'DISK_train'))
    os.mkdir(os.path.join(project_path, 'DISK_impute'))

    ### COPY EXAMPLE CONFIGS IN SUBDIRECTORY
    example_configs_folder = os.path.join(project_path, 'example_configs')
    os.mkdir(example_configs_folder)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    for config in ('config_create_project', 'config_prepare_data', 'config_train', 'config_test', 'config_impute'):
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


@config_reader(config_path="../conf/config_create_project.yaml")
def cli(_cfg) -> None:
    _cfg = parse_command_line_args(_cfg)

    for key in ('project_path', 'data_files', 'file_format'):
        val = _cfg.__dict__[key]
        if val is None or val == '_DEFAULT_':
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-create-project --project_path test_project --data_files x y '
                  f'z --file_format simple_csv\n')
            sys.exit(1)

    ## PROJECT_PATH
    if _cfg.project_path is None or type(_cfg.project_path) != str:
        print(f'\n❌ project_path should be string. Got {_cfg.project_path}.')
        sys.exit(1)
    else:
        if os.path.isabs(_cfg.project_path):
            if not os.path.exists(os.path.dirname(_cfg.project_path)):
                print(f'\n❌ project_path should be valid path. Got {_cfg.project_path}.')
                sys.exit(1)

    project_path = _cfg.project_path
    ext_project_path = 1
    final_project_path = str(project_path)
    while os.path.exists(final_project_path):
        final_project_path = project_path + f'_{ext_project_path}'
        ext_project_path += 1

    if final_project_path != project_path:
        print(f'⚠️️ A DISK project with same name {project_path} has been found.\n'
              f'Writing in {final_project_path} instead.\n')

    if not os.path.isabs(final_project_path):
        final_project_path = os.path.join(os.getcwd(), final_project_path)


    ## FILE_FORMAT
    if not check_file_type(_cfg.file_format):
        print(f'\n❌ File_type {_cfg.file_format} is not correct. Should be one '
              f'of {possible_file_type_values}.\n')
        sys.exit(1)

    file_format = _cfg.file_format

    ## FILE LIST
    if _cfg.data_files is None or type(_cfg.data_files) != list:
        print(f'\n❌ data_files should be a list of valid paths. '
              f'Got {_cfg.data_files}.')
        sys.exit(1)

    data_files = []
    ## CHECK FILE VALIDITY
    for f in _cfg.data_files:
        if os.path.isdir(f):
            possible_files = glob(os.path.join(f, '*'))
            for ff in possible_files:
                if not os.path.exists(ff):
                    print(f'\n❌ File {ff} not found. Please check path.\n')
                    continue

                if not check_extension(ff, file_format):
                    print(f'\n❌ File {ff} does not have the correct extension. '
                          f'Should be of type {file_format}.\n')
                    continue
                data_files.append(ff)
        else:
            if not os.path.exists(f):
                print(f'\n❌ File {f} not found. Please check path.\n')
                continue

            if not check_extension(f, file_format):
                print(f'\n❌ File {f} does not have the correct extension. '
                      f'Should be of type {file_format}.\n')
                continue

            data_files.append(f)

    if len(data_files) == 0:
        print(f'\n❌ No data files found. Please check given paths. \n')
        sys.exit(1)

    main(
        project_path=final_project_path,
        data_file_list=data_files,
        file_type=file_format,
    )


if __name__ == '__main__':
    cli()
    """
    # DISK syntax:
    DISK-create-project --project_path test_project --data_files x y z --file_format csv
    # careful no space between input_files inside the brackets, and after/before '='
    """