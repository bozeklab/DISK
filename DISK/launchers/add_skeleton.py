import sys
import os

import matplotlib.pyplot as plt
import yaml
import numpy as np
import networkx as nx

from DISK.utils.config_decorator import config_reader, parse_command_line_args

possible_colors = ['orange', 'cornflowerblue', 'hotpink', 'seagreen', 'gold', 'turquoise',  'purple', 'blue',
                   'darkolivegreen', 'grey', ]

def plot_skeleton(keypoints, neighbor_links, output_path) -> str:
    adj_matrix = np.zeros((len(keypoints), len(keypoints)))
    edge_color_dict = {}
    i_color = 0 % (len(possible_colors))
    for n in neighbor_links:
        if type(n[0]) == list:
            for nn in n:
                adj_matrix[nn[0], nn[1]] = 1
                adj_matrix[nn[1], nn[0]] = 1
                edge_color_dict[(keypoints[nn[0]], keypoints[nn[1]])] = possible_colors[i_color]
                edge_color_dict[(keypoints[nn[1]], keypoints[nn[0]])] = possible_colors[i_color]
        else:
            adj_matrix[n[0], n[1]] = 1
            adj_matrix[n[1], n[0]] = 1
            edge_color_dict[(keypoints[n[0]], keypoints[n[1]])] = possible_colors[i_color]
            edge_color_dict[(keypoints[n[1]], keypoints[n[0]])] = possible_colors[i_color]
        i_color += 1
        i_color %= len(possible_colors)
    G = nx.from_numpy_array(adj_matrix)

    node_names = {i: keypoints[i] for i in range(len(keypoints))}
    G = nx.relabel_nodes(G, node_names)

    edge_colors = []
    for edge in G.edges():
        edge_colors.append(edge_color_dict[edge])  # Color for Node A to Node B

    nx.draw(G, with_labels=True, node_color='lightgrey', node_size=700, font_size=15,
            edge_color=edge_colors)
    plt.suptitle('Random layout - does not represent coordinates')
    output_file = os.path.join(output_path, 'skeleton.png')
    plt.savefig(output_file)
    plt.close()

    return output_file

def check_input_group(group, list_index_keypoints):
    if len(group) > 2:
        return False
    for g in group:
        if type(g) != int or g not in list_index_keypoints:
            return False
    return True

def create_skeleton(keypoints: list):

    print('The keypoints are:')
    [print(f'{"":>11}{i} - {keypoints[i]}') for i in range(len(keypoints))]
    print('*' * 60, 'Please indicate the links between keypoints.',
          "You can either use one color per link, then write the links as follows:",
          "0, 1 <Enter> 2, 0",
          "Or you want to group links per color, then write as follows:",
          "(0, 1), (1, 2) <Enter> (3, 4), (4, 5)",
          "Just press <Enter> if no more links.", '*' * 60, sep='\n')
    neighbor_links = []
    link_colors = []
    i = 0
    while True:
        groups_of_neighbors = input("\nInput:  ")
        if groups_of_neighbors == '':
            break
        try:
            group = eval(groups_of_neighbors)
        except NameError:
            print('Wrong input. Please type again.')
            continue
        if type(group) == int or len(group) < 2:
            print('Wrong input. Please type again.')
            continue
        if type(group[0]) == int:
            group = list(group)
            if not check_input_group(group, list_index_keypoints=np.arange(len(keypoints))):
                print('Wrong input. Please type again.')
                continue
            neighbor_links.append(group)
        else:
            final_group = []
            for g in group:
                list_g = list(g)
                if not check_input_group(list_g, list_index_keypoints=np.arange(len(keypoints))):
                    print('Wrong input. Please type again.')
                    continue
                final_group.append(list_g)
            neighbor_links.append(final_group)
        link_colors.append(possible_colors[i % len(possible_colors)])
        i += 1

    center = None
    while center is None:
        center_index = input("\nIndicate which keypoint index is closer to the center of mass of the animal. "
                             "Please pick only one index. Should be an integer. ")
        try:
            center = int(center_index)
        except NameError:
            print('Wrong input. Please type again.')

        if center not in np.arange(len(keypoints)):
            print('Wrong input. Please type again.')
            center = None

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

    output_file = plot_skeleton(keypoints, neighbor_links, config['project_path'])

    ### CREATE DISK_PROJECT_LOG
    updated_config = config
    updated_config['skeleton_center'] = center
    updated_config['skeleton'] = neighbor_links
    updated_config['skeleton_colors'] = link_colors

    # Save the default configuration to a YAML file
    with open(config_path, 'w') as file:
        yaml.dump(updated_config, file)

    print(f'✅ Successfully added skeleton. A visualization is available at {output_file}.\n')

    return


if __name__ == '__main__':
    cli()
    """
    # DISK syntax:
    DISK-add-skeleton --project_name path/to/my/project
    """