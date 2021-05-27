import csv


def parse(path):
    data = []
    with open(path, newline='') as dataset:
        reader = csv.DictReader(dataset)
        for element in reader:
            data.append(element)
    return data
