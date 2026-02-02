

def main():

    try:
        import DISK
        import hydra
        from omegaconf import DictConfig

        @hydra.main(version_base=None, config_path="../conf", config_name="conf_missing")
        def test_main(_cfg: DictConfig):
            print(_cfg)

        test_main()
        print("✅ DISK is installed successfully.")
    except Exception as e:
        print("❌ There is a problem with DISK installation."
              "\nPlease retry from scratch (conda env) and post a Github issue")

    import torch
    if torch.cuda.is_available():
        print("✅ GPU is available")
    else:
        print("⚠️ GPU is not found.")

def cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Verifies that DISK was installed successfully."
    )
    args = parser.parse_args()

    main()


if __name__ == '__main__':
    cli()