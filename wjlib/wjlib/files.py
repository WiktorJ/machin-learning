import csv


def read_file_to_tuples(path, delimiter=" "):
    print('opening file', path)
    with open(path) as the_file:
        return [tuple(line) for line in csv.reader(the_file, delimiter=delimiter)]