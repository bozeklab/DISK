

def main():

    try:
        import DISK
        version = DISK._version.version
        print("DISK Version Found:", version)

        from DISK.utils.config_decorator import config_reader

        @config_reader(config_path="../conf/config_create_project.yaml")
        def test_main(_cfg):
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
