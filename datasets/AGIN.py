import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
import csv


@DATASET_REGISTRY.register()
class AGIN(DatasetBase):

    dataset_dir = '/hd2/wangzichuan/AGIN'

    def __init__(self, cfg):
        self.root = '/hd2/wangzichuan/AGIN'
        self.train_data, self.val_data, self.test_data = self._load_data()
        # train_u
        super().__init__(train_x=self.train_data, val=self.val_data, test=self.test_data)
        # super().__init__(train_x=self.train_data, test=self.test_data)

    def _load_data(self):
        csv_file = self.root + '/' + 'MOS.csv'
        data = pd.read_csv(csv_file)

        # Split data into train and test
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Split train data into train and validation
        train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=42)

        train_dataset = self._process_data(train_data, 'train')
        val_dataset = self._process_data(val_data, 'val')
        test_dataset = self._process_data(test_data, 'test')

        return train_dataset, val_dataset, test_dataset

    def _process_data(self, data, csv_name):
        dataset = []
        csv_filename = '/hd2/wangzichuan/CoOp/datasets/AGIN/' + csv_name + '.csv'
        fieldnames = ['impath', 'mos', 'label', 'classname']
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            for index, row in data.iterrows():
                impath = os.path.join(self.root+'/images', row["filename"])
                mos = float(row["MOS_nat"])
                if 0 <= mos < 0.5:
                    label = 0
                    classname = 'terrible'
                elif 0.5 <= mos < 1.5:
                    label = 1
                    classname = 'bad'
                elif 1.5 <= mos < 2.5:
                    label = 2
                    classname = 'poor'
                elif 2.5 <= mos < 3.5:
                    label = 3
                    classname = 'average'
                elif 3.5 <= mos < 4.5:
                    label = 4
                    classname = 'good'
                else:
                    label = 5
                    classname = 'perfect'
                # if 0 <= mos_qual < 1.231:
                #     label = 0
                #     classname = 'terrible'
                # elif 1.231 <= mos_qual < 2.266:
                #     label = 1
                #     classname = 'bad'
                # elif 2.266 <= mos_qual < 2.749:
                #     label = 2
                #     classname = 'poor'
                # elif 2.749 <= mos_qual < 3.092:
                #     label = 3
                #     classname = 'average'
                # elif 3.092 <= mos_qual < 3.456:
                #     label = 4
                #     classname = 'good'
                # else:
                #     label = 5
                #     classname = 'perfect'
                writer.writerow([impath, mos, label, classname])
                dataset.append(Datum(impath=impath, label=label, classname=classname, mos=mos))

                # mos_qual = float(row["mos_align"])
                # prompt = row["prompt"]
                # if 0 <= mos_qual < 0.5:
                #     label = 0
                #     classname = prompt
                # elif 0.5 <= mos_qual < 1.5:
                #     label = 1
                #     classname = prompt
                # elif 1.5 <= mos_qual < 2.5:
                #     label = 2
                #     classname = prompt
                # elif 2.5 <= mos_qual < 3.5:
                #     label = 3
                #     classname = prompt
                # elif 3.5 <= mos_qual < 4.5:
                #     label = 4
                #     classname = prompt
                # else:
                #     label = 5
                #     classname = prompt
                # label = 0
                # classname = prompt
                # writer.writerow([impath, mos_qual, label, classname])
                # dataset.append(Datum(impath=impath, label=label, classname=classname, mos=mos_qual))
                # print(dataset)

        return dataset

    # def __len__(self):
    #     if self.split == "train":
    #         print(len(self.train_data))
    #         return len(self.train_data)
    #     elif self.split == "val":
    #         return len(self.val_data)
    #     elif self.split == "test":
    #         return len(self.test_data)

    # def __getitem__(self, idx):
    #     if self.split == "train":
    #         return self.train_data[idx]
    #     elif self.split == "val":
    #         return self.val_data[idx]
    #     elif self.split == "test":
    #         return self.test_data[idx]