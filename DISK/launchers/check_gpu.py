

def main():

    import torch
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        print("GPU Found!")
    else:
        print("GPU not found! \nPlease reinstall torch linked to your Cuda wheels (details in DISK readme.")




def cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Check if the GPU is working"
    )

    parser.parse_args()

    main()



if __name__ == '__main__':
    cli()