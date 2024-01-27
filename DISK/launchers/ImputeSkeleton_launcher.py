#!/usr/bin/env python

import DISK
import argparse
import os
import sys
import logging

def main():

    LAUNCHER_DIR = os.path.join(os.path.dirname(__file__))
    PYCKAGE_RESOURCES_DIR = os.path.join(os.path.abspath(os.path.join(LAUNCHER_DIR,os.pardir)),"resources")

    parser = argparse.ArgumentParser()

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

    show_progress_bar = False
    if args.progress_bar:
        show_progress_bar = True

    DISK.app_module1.appfunction_of_module1(loglevel=verboselevel,
                                                      progress_bar=show_progress_bar,
                                                      resources=PYCKAGE_RESOURCES_DIR, args=vars(args))

if __name__ == "__main__":
    main()
