

def main():

    try:
        import DISK
        version = DISK._version.version
        print("DISK Version Found:", version)

        import hydra
        from omegaconf import DictConfig

        @hydra.main(version_base=None, config_path="../conf", config_name="config_create_project")
        def test_main(_cfg: DictConfig):
            _cfg.keys()

        test_main()
        print("✅ DISK is installed successfully.")
    except Exception:
        print("❌ There is a problem with DISK installation."
              "\n  Please retry from scratch (conda env) and post a Github issue")

    try:
        import torch
        if torch.cuda.is_available():
            print("✅ GPU is available")
        else:
            print("⚠️ GPU is not found.")
    except Exception:
        print("❌ There is a problem with Torch installation."
              "\n  Please retry installing pytorch (see README for details).")


def cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Verifies that DISK was installed successfully."
    )
    args = parser.parse_args()

    main()


if __name__ == '__main__':
    cli()