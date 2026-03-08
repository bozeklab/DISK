from DISK.utils.config_decorator import config_reader, parse_command_line_args


# Example function that uses the configuration
@config_reader(config_path='../conf/config_create_project.yaml')  # Specify your relative path here
def run_application(config):
    args = parse_command_line_args(config)
    print(args)

    # Override config values with command line arguments if provided
    param1 = args.dataset_name
    param2 = args.freq_original_freq


    print(f'Parameter 1: {param1}')
    print(f'Parameter 2: {param2} {type(param2)}')

# Run the application
if __name__ == '__main__':
    run_application()