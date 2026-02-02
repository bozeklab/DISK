#!/usr/bin/env python
import argparse
import logging
import os
import sys


def main(loglevel, progress_bar,resources,args):
    logging.info('[INFO] Welcome in DISK')
    logging.debug('[DEBUG] Now in main() : {}'.format(main))
    logging.debug('[DEBUG] Parsed arguments = {}'.format(args))
    logging.debug('[DEBUG] loglevel : {}, progress_bar : {}, resources located in {}'.format(loglevel,progress_bar,resources))
    return


def cli():

    LAUNCHER_DIR = os.path.join(os.path.dirname(__file__))
    PYCKAGE_RESOURCES_DIR = os.path.join(os.path.abspath(os.path.join(LAUNCHER_DIR,os.pardir)),"resources")

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbosity',"-v",
                            help='Choose your verbosity. Default: INFO',
                            required=False,
                            default="DEBUG",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    parser.add_argument('--progress_bar',"-p",
                            help='Displays a progress bar',
                            action='store_true')


    args = parser.parse_args()


    verboselevel = "logging."+str(args.verbosity)
    logging.basicConfig(level=eval(verboselevel),
                        format='%(asctime)s %(message)s',
                        stream=sys.stdout)


    main(
        loglevel=verboselevel,
        progress_bar=args.progress_bar,
        resources=PYCKAGE_RESOURCES_DIR,
        args=vars(args),
    )

if __name__ == "__main__":
    cli()
