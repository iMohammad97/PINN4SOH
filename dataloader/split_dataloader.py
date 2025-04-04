import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import random
from sklearn.model_selection import train_test_split
from utils.util import write_to_txt

class DF():
    def __init__(self,args):
        self.normalization = True
        self.normalization_method = args.normalization_method # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        '''
        :param Ser1:
        :return: index
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self,df):
        '''
        :param df: DataFrame
        :return: DataFrame
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df

    def read_one_csv(self,file_name,nominal_capacity=None):
        '''
        read a csv file and return a DataFrame
        :param file_name: str
        :return: DataFrame
        '''
        df = pd.read_csv(file_name)
        df.insert(df.shape[1]-1,'cycle index',np.arange(df.shape[0]))

        df = self.delete_3_sigma(df)

        if nominal_capacity is not None:
            #print(f'nominal_capacity:{nominal_capacity}, capacity max:{df["capacity"].max()}',end=',')
            df['capacity'] = df['capacity']/nominal_capacity
            #print(f'SOH max:{df["capacity"].max()}')
            f_df = df.iloc[:,:-1]
            if self.normalization_method == 'min-max':
                f_df = 2*(f_df - f_df.min())/(f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean())/f_df.std()

            df.iloc[:,:-1] = f_df

        return df

    def load_one_battery(self,path,nominal_capacity=None):
        '''
        Read a csv file and divide the data into x and y
        :param path:
        :param nominal_capacity:
        :return:
        '''
        df = self.read_one_csv(path,nominal_capacity)
        x = df.iloc[:,:-1].values
        y = df.iloc[:,-1].values
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        return (x1,y1),(x2,y2)

    # MY CHANGES: New helper function for loading & concatenating battery data from a list of file paths
    def _load_and_concat(self, path_list, nominal_capacity):
        '''
        Load data from multiple battery CSVs and concatenate them into single arrays.
        This is used internally for battery-level splitting.
        '''
        # MY CHANGES: We moved this logic from "load_all_battery" into a separate function.
        X1, X2, Y1, Y2 = [], [], [], []
        for path in path_list:
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)

        return X1, X2, Y1, Y2

    def load_all_battery(self,path_list,nominal_capacity):
        '''
        Read multiple csv files, and package them into dataloaders.
        By default, this function now ensures *battery-level* splitting:
        1) 80% of batteries for training/validation, 20% of batteries for testing.
        2) Then, from the training subset, 80% for training, 20% for validation.

        # MY CHANGES: Instead of randomly splitting *rows* across all batteries,
        # we now split *battery file paths* so that entire batteries are reserved
        # either for training, validation, or testing.
        '''

        # ------------------------------------------------------------------
        # OLD CODE (Commented out):
        #
        # X1, X2, Y1, Y2 = [], [], [], []
        # for path in path_list:
        #     (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
        #     X1.append(x1)
        #     X2.append(x2)
        #     Y1.append(y1)
        #     Y2.append(y2)
        #
        # X1 = np.concatenate(X1, axis=0)
        # X2 = np.concatenate(X2, axis=0)
        # Y1 = np.concatenate(Y1, axis=0)
        # Y2 = np.concatenate(Y2, axis=0)
        #
        # # Then the code did row-level splits (80-20, etc.)
        # ...
        # ------------------------------------------------------------------

        # MY CHANGES: Start of new battery-level approach
        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder,self.args.log_dir)
            write_to_txt(save_name,'data path:')
            write_to_txt(save_name,str(path_list))

        # 1) Randomly split the entire path_list (each file is presumably one battery)
        #    into train+valid and test sets (80%-20%).
        train_valid_paths, test_paths = train_test_split(path_list, test_size=0.2, random_state=420)

        # 2) From the train_valid_paths, split again for training and validation (80%-20%).
        if len(train_valid_paths) > 1:
            train_paths, valid_paths = train_test_split(train_valid_paths, test_size=0.2, random_state=420)
        else:
            # Edge case: if there's only 1 battery in train_valid, we treat it all as train
            train_paths = train_valid_paths
            valid_paths = []

        # 3) Load and concatenate data from each subset.
        X1_train, X2_train, Y1_train, Y2_train = self._load_and_concat(train_paths, nominal_capacity) \
            if len(train_paths) > 0 else ([], [], [], [])
        X1_valid, X2_valid, Y1_valid, Y2_valid = self._load_and_concat(valid_paths, nominal_capacity) \
            if len(valid_paths) > 0 else ([], [], [], [])
        X1_test,  X2_test,  Y1_test,  Y2_test  = self._load_and_concat(test_paths,  nominal_capacity) \
            if len(test_paths) > 0  else ([], [], [], [])

        # Convert lists to numpy arrays if they aren't empty
        if len(X1_train) != 0: X1_train, X2_train, Y1_train, Y2_train = map(np.array, [X1_train, X2_train, Y1_train, Y2_train])
        if len(X1_valid) != 0: X1_valid, X2_valid, Y1_valid, Y2_valid = map(np.array, [X1_valid, X2_valid, Y1_valid, Y2_valid])
        if len(X1_test)  != 0: X1_test,  X2_test,  Y1_test,  Y2_test  = map(np.array, [X1_test, X2_test, Y1_test, Y2_test])

        # 4) Convert to torch tensors (only if sets are non-empty)
        if len(X1_train) != 0:
            tensor_X1_train = torch.from_numpy(X1_train).float()
            tensor_X2_train = torch.from_numpy(X2_train).float()
            tensor_Y1_train = torch.from_numpy(Y1_train).float().view(-1,1)
            tensor_Y2_train = torch.from_numpy(Y2_train).float().view(-1,1)
            train_loader = DataLoader(TensorDataset(tensor_X1_train, tensor_X2_train, tensor_Y1_train, tensor_Y2_train),
                                      batch_size=self.args.batch_size, shuffle=True)
        else:
            train_loader = None

        if len(X1_valid) != 0:
            tensor_X1_valid = torch.from_numpy(X1_valid).float()
            tensor_X2_valid = torch.from_numpy(X2_valid).float()
            tensor_Y1_valid = torch.from_numpy(Y1_valid).float().view(-1,1)
            tensor_Y2_valid = torch.from_numpy(Y2_valid).float().view(-1,1)
            valid_loader = DataLoader(TensorDataset(tensor_X1_valid, tensor_X2_valid, tensor_Y1_valid, tensor_Y2_valid),
                                      batch_size=self.args.batch_size, shuffle=True)
        else:
            valid_loader = None

        if len(X1_test) != 0:
            tensor_X1_test = torch.from_numpy(X1_test).float()
            tensor_X2_test = torch.from_numpy(X2_test).float()
            tensor_Y1_test = torch.from_numpy(Y1_test).float().view(-1,1)
            tensor_Y2_test = torch.from_numpy(Y2_test).float().view(-1,1)
            test_loader = DataLoader(TensorDataset(tensor_X1_test, tensor_X2_test, tensor_Y1_test, tensor_Y2_test),
                                     batch_size=self.args.batch_size, shuffle=False)
        else:
            test_loader = None

        # 5) Build return dictionary
        loader = {
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader
        }

        # MY CHANGES: We skip the old Condition 1, 2, 3 row-based splits
        # and only return the new battery-level splits (train, valid, test).
        return loader
        # MY CHANGES: End of new battery-level approach


class XJTUdata(DF):
    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.file_list = os.listdir(root)
        self.variables = pd.read_csv(os.path.join(root, self.file_list[0])).columns
        self.num = len(self.file_list)
        self.batch_names = ['2C','3C','R2.5','R3','RW','satellite']
        self.batch_size = args.batch_size

        if self.normalization:
            self.nominal_capacity = 2.0
        else:
            self.nominal_capacity = None
        #print('-'*20,'XJTU data','-'*20)

    def read_one_batch(self,batch='2C'):
        '''
        读取一个批次的csv文件,并把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version: Read a batch of csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :param batch: int or str:batch
        :return: dict
        '''
        if isinstance(batch,int):
            batch = self.batch_names[batch]
        assert batch in self.batch_names, 'batch must be in {}'.format(self.batch_names)
        file_list = []
        for i in range(self.num):
            if batch in self.file_list[i]:
                path = os.path.join(self.root,self.file_list[i])
                file_list.append(path)
        return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件，并把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version: Read all csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for file in self.file_list:
                path = os.path.join(self.root, file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list,nominal_capacity=self.nominal_capacity)


class HUSTdata(DF):
    def __init__(self,root='../data/HUST data',args=None):
        super(HUSTdata, self).__init__(args)
        self.root = root
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-'*20,'HUST data','-'*20)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件:如果指定了specific_path_list,则读取指定的文件；否则读取所有文件；
        把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version:
        Read all csv files.
        If specific_path_list is not None, read the specified file;
        otherwise read all files;
        :param self:
        :param specific_path:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            files = os.listdir(self.root)
            for file in files:
                path = os.path.join(self.root,file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)


class MITdata(DF):
    def __init__(self,root='../data/MIT data',args=None):
        super(MITdata, self).__init__(args)
        self.root = root
        self.batchs = ['2017-05-12','2017-06-30','2018-04-12']
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-' * 20, 'MIT data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        读取一个批次的csv文件
        English version: Read a batch of csv files
        :param batch: int,可选[1,2,3]
        :return: dict
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件。如果指定了specific_path_list,则读取指定的文件；否则读取所有文件；封装成dataloader
        English version:
        Read all csv files.
        If specific_path_list is not None, read the specified file; otherwise read all files;
        :param self:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for batch in self.batchs:
                root = os.path.join(self.root,batch)
                files = os.listdir(root)
                for file in files:
                    path = os.path.join(root,file)
                    file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)

class TJUdata(DF):
    def __init__(self,root='../data/TJU data',args=None):
        super(TJUdata, self).__init__(args)
        self.root = root
        self.batchs = ['Dataset_1_NCA_battery','Dataset_2_NCM_battery','Dataset_3_NCM_NCA_battery']
        if self.normalization:
            self.nominal_capacities = [3.5,3.5,2.5]
        else:
            self.nominal_capacities = [None,None,None]
        #print('-' * 20, 'TJU data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        读取一个批次的csv文件
        English version: Read a batch of csv files
        :param batch: int,可选[1,2,3]; optional[1,2,3]
        :return: DataFrame
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacities[batch])

    def read_all(self,specific_path_list):
        '''
        读取所有csv文件,封装成dataloader
        English version: Read all csv files and encapsulate them into a dataloader
        :param self:
        :return: dict
        '''
        for i,batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)


if __name__ == '__main__':
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data',type=str,default='MIT',help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch',type=int,default=1,help='1,2,3')
        parser.add_argument('--batch_size',type=int,default=256,help='batch size')
        parser.add_argument('--normalization_method',type=str,default='z-score',help='min-max,z-score')
        parser.add_argument('--log_dir',type=str,default='test.txt',help='log dir')
        parser.add_argument('--save_folder',type=str,default=None,help='folder to save logs')
        return parser.parse_args()

    args = get_args()

    # Example usage with MIT data
    mit = MITdata(args=args)
    loader = mit.read_all()

    train_loader = loader['train']
    valid_loader = loader['valid']
    test_loader = loader['test']

    print('train_loader:',
          len(train_loader) if train_loader else 0,
          'valid_loader:',
          len(valid_loader) if valid_loader else 0,
          'test_loader:',
          len(test_loader) if test_loader else 0)

    # Example iteration over train_loader (if not None)
    if train_loader is not None:
        for i, (x1, x2, y1, y2) in enumerate(train_loader):
            print('Batch:', i)
            print('x1 shape:', x1.shape)
            print('x2 shape:', x2.shape)
            print('y1 shape:', y1.shape)
            print('y2 shape:', y2.shape)
            break
