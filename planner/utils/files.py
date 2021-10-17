import pathlib

resources = pathlib.Path(__file__).parent / '../../resources'

if __name__ == '__main__':
    print(list(resources.glob('*')))
