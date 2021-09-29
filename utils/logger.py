import csv
import os

class Logger:
    def __init__(self, filepath='./', filename='results.csv'):
        if not os.path.exists(filepath): os.makedirs(filepath)
        self.csv_file_path = os.path.join(filepath, filename)

    def write(self, data_dict):
        """warning: this allows for wrong keys to be passed"""
        if os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'a', newline='') as f: #newline='' is to make it windows compatible
                writer = csv.writer(f)
                writer.writerow(list(data_dict.values()))
        else:
            with open(self.csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(data_dict.keys()))
                writer.writerow(list(data_dict.values()))



if __name__=="__main__":
    logger = Logger('./../logs/', 'test.csv')
    data1 = {'test1': 5, 'test2':49}
    logger.write(data1)
    data2 = {'test1': 55, 'test2':4949}
    logger.write(data2)
    import pandas as pd
    print(pd.read_csv("./../logs/test.csv"))





