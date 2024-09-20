import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
import csv


@DATASET_REGISTRY.register()
class AIGCIQA2023(DatasetBase):

    dataset_dir = '/hd2/wangzichuan/AIGCIQA2023/allimg'

    def __init__(self, cfg):
        self.root = '/hd2/wangzichuan/AIGCIQA2023'
        # self.train_data, self.val_data, self.test_data = self._load_data()
        self.train_data, self.test_data = self._load_data()
        # train_u
        # super().__init__(train_x=self.train_data, val=self.val_data, test=self.test_data)
        super().__init__(train_x=self.train_data, test=self.test_data)


    def _load_data(self):
        # csv_file = self.root + '/' + 'mos_' + 'quality' + '.csv' 
        csv_file = self.root + '/' + 'mos_' + 'authenticity' + '.csv' 
        
        if 'quality' in csv_file:
            eval = 'quality'
        elif 'authenticity' in csv_file:
            eval = 'authenticity'
        data = pd.read_csv(csv_file)

        # Split data into train and test
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Split train data into train and validation
        # train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

        train_dataset = self._process_data(train_data, eval, 'train')
        # val_dataset = self._process_data(val_data, 'val')
        test_dataset = self._process_data(test_data, eval, 'test')

        # return train_dataset, val_dataset, test_dataset
        return train_dataset, test_dataset

    def _process_data(self, data, eval, csv_name):
        dataset = []
        csv_filename = '/hd2/wangzichuan/CLIP-AGIQA/datasets/AIGCIQA2023/' + eval + '/' + csv_name + '.csv'
        fieldnames = ['impath', 'mos', 'label', 'classname']
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            for index, row in data.iterrows():
                imgpath = self.root + '/' + "allimg"
                impath = os.path.join(imgpath, row["name"])
                mos_qual = float(row["mos"])
                if 0 <= mos_qual < 30:
                    label = 0
                    classname = 'terrible'
                elif 30 <= mos_qual < 40:
                    label = 1
                    classname = 'bad'
                elif 40 <= mos_qual < 50:
                    label = 2
                    classname = 'poor'
                elif 50 <= mos_qual < 60:
                    label = 3
                    classname = 'average'
                elif 60 <= mos_qual < 70:
                    label = 4
                    classname = 'good'
                else:
                    label = 5
                    classname = 'perfect'
                writer.writerow([impath, mos_qual, label, classname])
                dataset.append(Datum(impath=impath, label=label, classname=classname, mos=mos_qual))


        return dataset
