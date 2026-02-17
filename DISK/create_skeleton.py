create_skeleton_file = input('Would you like to create a skeleton file [y/n]? \n'
                             '(If this is the first time creating a dataset for a specific data, then type y. \n'
                             'If a skeleton file has already been generated for this type of data (animal + recording type), then type n. ')

possible_colors = ['orange', 'gold', 'grey', 'cornflowerblue', 'turquoise', 'hotpink', 'purple', 'blue', 'seagreen',
                   'darkolivegreen', ]

if create_skeleton_file.lower() == 'y':  ## answer is yes, create a skeleton file
    print('The keypoints are:')
    [print(f'{"":>11}{i} - {keypoints[i]}') for i in range(len(keypoints))]
    print('Please indicate the links between keypoints (if possible in groups of links,\n'
          'e.g. a leg, or the spine - groups of links will be displayed in the same color. ')
    neighbor_links = []
    link_colors = []
    i = 0
    while True:
        groups_of_neighbors = input("Indicate the first group using the keypoints' indices and "
                                    "follow the convention (0, 2), (0, 6), (2, 4) or (0, 2) \n"
                                    "for only one link in a group. "
                                    "Just press <Enter> if no more links. ")
        if groups_of_neighbors == '':
            break
        group = eval(groups_of_neighbors)
        neighbor_links.append(group)
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

    ## Now right the skeleton file
    skeleton_file_path = os.path.join(outputdir, 'skeleton.py')
    with open(skeleton_file_path, 'w') as opened_file:
        txt = f"num_keypoints = {len(keypoints)}\n"
        txt += f"keypoints = {keypoints}\n"
        # DIVIDER= 2 for 2D, 3 for 3D, sometimes additional dimension for a confidence score or an error
        # score for the detection
        txt += f"center = {center}\n"
        txt += f"original_directory = '{outputdir}'\n"
        txt += f"neighbor_links = {neighbor_links}\n"
        txt += f"link_colors = {link_colors}\n"

        opened_file.write(txt)