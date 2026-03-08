import sys
import os
import yaml

from DISK.utils.config_decorator import config_reader, parse_command_line_args

def create_skeleton(keypoints: list):

    possible_colors = ['orange', 'gold', 'grey', 'cornflowerblue', 'turquoise', 'hotpink', 'purple', 'blue', 'seagreen',
                       'darkolivegreen', ]

    print('The keypoints are:')
    [print(f'{"":>11}{i} - {keypoints[i]}') for i in range(len(keypoints))]
    print('Please indicate the links between keypoints (if possible in groups of links,\n'
          'e.g. a leg, or the spine - groups of links will be displayed in the same color. ')
    neighbor_links = []
    link_colors = []
    i = 0
    while True:
        groups_of_neighbors = input("You can either use one color per link, then write the links as follows:\n"
                                    "0, 1 <Enter> 2, 0\n"
                                    "Or you want to group links per color, then write as follows:\n"
                                    "(0, 1), (1, 2) <Enter> (3, 4), (4, 5)\n"
                                    "Just press <Enter> if no more links.\nYour answer:  ")
        if groups_of_neighbors == '':
            break
        group = eval(groups_of_neighbors)
        if type(group[0]) == int:
            neighbor_links.append(list(group))
        else:
            neighbor_links.append([list(g) for g in group])
        link_colors.append(possible_colors[i % len(possible_colors)])
        i += 1

    center = None
    while center is None:
        center_index = input("Indicate which keypoint index is closer to the center of mass of the animal. "
                             "Please pick only one index. Should be an integer. ")
        try:
            center = int(center_index)
        except NameError:
            print('Wrong input')

    return center, neighbor_links, link_colors



@config_reader(config_path="../conf/config_prepare_data.yaml")
def cli(_cfg) -> None:
    _cfg = parse_command_line_args(_cfg)

    for key in ('project_path', ):
        val = _cfg.__dict__[key]
        if val is None:
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-add-skeleton project_path=test_project\n')
            sys.exit(1)

    if _cfg.project_path is None or type(_cfg.project_path) != str:
        print("\n❌ project_path is a required parameter and should be a "
              "valid path to the config_project.yaml file. "
              f"Got {_cfg.project_path}")
        sys.exit(1)
    else:
        project_path = _cfg.project_path

    ### LOAD PROJECT LOG
    # Load the YAML configuration file
    config_path = os.path.join(project_path, 'config_project.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if 'keypoints' not in config.keys():
        print("\n❌ keypoint list not found in config file. "
              "Please run once prepare-data before add-skeleton\n")
        sys.exit(1)
    keypoints = config['keypoints']

    center, neighbor_links, link_colors = create_skeleton(keypoints)

    ### CREATE DISK_PROJECT_LOG
    updated_config = config
    updated_config['skeleton_center'] = center
    updated_config['skeleton'] = neighbor_links
    updated_config['skeleton_colors'] = link_colors

    # Save the default configuration to a YAML file
    with open(config_path, 'w') as file:
        yaml.dump(updated_config, file)


if __name__ == '__main__':
    cli()
    """
    # DISK syntax:
    DISK-create-project dir=mydir project_name=test_project input_files=[x,y,z] file_type=csv
    """