import pathlib

API_URL = 'https://api.jolpi.ca/ergast/f1'

DATA_FOLDER = pathlib.Path('data')
BIN_FOLDER = DATA_FOLDER / 'bin'
TRAIN_FOLDER = DATA_FOLDER / 'train'
TEST_FOLDER = DATA_FOLDER / 'test'
VALIDATION_FOLDER = DATA_FOLDER / 'val'

MODEL_FOLDER = pathlib.Path('models')

OUT_FOLDER = pathlib.Path('out')
