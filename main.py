import argparse
import logging

from src.app import start
from src.utils import ColourFormatter


def run(choice: int | None = None) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(ColourFormatter())
    logger.addHandler(handler)

    start(choice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='F1 Race Forecasting'
    )
    parser.add_argument(
        'choice',
        nargs='?',
        type=int,
        help='Menu Options: 1=Fetch, 2=Parse, 3=Predict, 4=Exit'
    )
    args = parser.parse_args()

    try:
        run(args.choice)
    except KeyboardInterrupt:
        pass
