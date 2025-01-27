import os 
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy

import config

warnings.filterwarnings('ignore')


class DataProcessor:
    def __init__(self, data_dir='OpportunityUCIDataset/dataset'):
        self.data_dir = data_dir
        self.columns = []
        self.data_collection = pd.DataFrame()
        self.column_to_sensor = {}
        self.sensor_to_column = {}
        self.track_dict = {}
        self.reset_label_to_original_label = {}
        self.read_colum_names()
        self.init_label_legend()
        self.sensor_to_position = config.sensor_to_position
        self.position_to_sensor = config.position_to_sensor
        self.position_to_original_position_label = config.position_to_original_position_label

    def get_index(self, string):
        """
        Helper function to extract the index from a line.
        get the index of the first alphabet in the string after the 9th index

        Args:
        line (str): Input line containing index information.

        Returns:
        int: Extracted index.
        """

        for i in range(9,len(string)):
            if string[i].isalpha():
                return i
        return -1
    
    def read_colum_names(self):
        with open(os.path.join(self.data_dir, "column_names.txt"), 'r') as f:
            lines = f.read().splitlines()

            for line in lines:
                if 'Column' in line:
                    # Extract column names and append to the list
                    column_name = line[self.get_index(line):].split(";")[0]
                    
                    if " " in column_name:
                        sensor_name = column_name.split(" ")[1]
                        self.column_to_sensor[column_name] = sensor_name
                        if sensor_name in self.sensor_to_column:
                            self.sensor_to_column[sensor_name].append(column_name)
                        else:
                            self.sensor_to_column[sensor_name] = [column_name]
                    self.columns.append(column_name)

    def extract_train_data(self, test_file="S4-ADL5.dat"):
        """
        Extracts traindata from .dat files in the OpportunityUCIDataset/dataset folder.

        Returns:
        pandas.DataFrame: Dataframe containing extracted data.
        """

        # Get all the .dat files in the dataset folder
        files = os.listdir(self.data_dir)
        # files = [f for f in files if f.endswith('.dat')]
        files = [f for f in files if f.startswith("S4") and f.endswith('.dat')]

        # Separate the ADL and Drill files
        list_of_files = [f for f in files if 'Drill' not in f]

        # Remove the test file from the list
        list_of_files.remove(test_file)

        # Create an empty DataFrame with the extracted column names
        self.data_collection = pd.DataFrame(columns=self.columns)

        # Iterate over the list of files and concatenate data to the DataFrame
        for _, file in enumerate(sorted(list_of_files)):
            proc_data = pd.read_table(os.path.join(self.data_dir, file), header=None, sep='\s+')
            print(file, len(proc_data))
            proc_data.columns = self.columns
            self.data_collection = pd.concat([self.data_collection, proc_data])

        # Reset the DataFrame index
        self.data_collection.reset_index(drop=True, inplace=True)

    def extract_test_data(self, test_file="S4-ADL5.dat"):
        """
        Extracts testdata from .dat files in the OpportunityUCIDataset/dataset folder.

        Returns:
        pandas.DataFrame: Dataframe containing extracted data.
        """
        # Create an empty DataFrame with the extracted column names
        self.data_collection = pd.DataFrame(columns=self.columns)

        proc_data = pd.read_table(os.path.join(self.data_dir, test_file), header=None, sep='\s+')
        print(test_file, len(proc_data))
        proc_data.columns = self.columns
        self.data_collection = pd.concat([self.data_collection, proc_data])

        self.data_collection.reset_index(drop=True, inplace=True)

    def data_cleaning(self):
        """
        Performs data cleaning on the input DataFrame.

        Args:
        data_collection (pandas.DataFrame): Input DataFrame.

        Returns:
        pandas.DataFrame: Cleaned DataFrame.
        """

        # Drop columns with more than 10% NaN values
        threshold = int(len(self.data_collection.columns) * 0.9)
        self.data_collection = self.data_collection.dropna(thresh=threshold, inplace=False)

        # Convert non-numeric data to NaN
        self.data_collection = self.data_collection.apply(pd.to_numeric, errors='coerce')

        # Fill NaN values using an interpolation method
        self.data_collection = self.data_collection.interpolate()

        # Drop any remaining NaN values to ensure only non-NaN data remains
        self.data_collection = self.data_collection.dropna()

    def init_label_legend(self):
        """
        Resets labels in the given DataFrame based on the information from 'label_legend.txt'.
        """

        # Read label_legend.txt file
        labels = pd.read_csv('OpportunityUCIDataset/dataset/label_legend.txt', sep='   -   ', header=0)

        for track in labels['Track name'].unique():
            self.track_dict[track] = dict(labels.loc[labels['Track name'] == track][["Unique index", "Label name"]].to_numpy())

        # Special case for 'Locomotion' track
        for track in self.track_dict:
            if track == 'Locomotion':
                self.track_dict[track][1] = 1
                self.track_dict[track][2] = 2
                self.track_dict[track][4] = 3
                self.track_dict[track][5] = 4
            else:
                i = 1
                for key in self.track_dict[track]:
                    self.track_dict[track][key] = i
                    i += 1

        for track in self.track_dict:
            self.reset_label_to_original_label[track] = {}
            self.reset_label_to_original_label[track][0] = 0
            for key in self.track_dict[track]:
                original_label = key
                reset_label = self.track_dict[track][key]
                self.reset_label_to_original_label[track][reset_label] = int(original_label)

    def reset_label(self):
        """
        Resets labels in the given DataFrame based on the information from 'label_legend.txt'.

        Args:
        data_collection (pandas.DataFrame): Input DataFrame.

        Returns:
        pandas.DataFrame: DataFrame with reset labels.
        """
        # Update labels in the DataFrame based on the mapping
        for track in self.track_dict:
            for key in self.track_dict[track]:
                self.data_collection.loc[self.data_collection[track] == key, track] = self.track_dict[track][key]

    def normalize_data(self):
        """
        Normalize numeric columns in the DataFrame using StandardScaler. 
        Except for the last 7 columns(activity labels) and the first column(milisecond).

        Args:
        df (pandas.DataFrame): Input DataFrame.

        Returns:
        pandas.DataFrame: DataFrame with normalized numeric columns.
        """
        
        scaler = StandardScaler()
        self.data_collection[self.data_collection.columns[1:-7]] = scaler.fit_transform(self.data_collection[self.data_collection.columns[1:-7]])

    def read_data(self, testset=False):
        if testset:
            return pd.read_csv('OpportunityUCIDataset/dataset/cleaned_testdata.csv')
        return pd.read_csv('OpportunityUCIDataset/dataset/cleaned_traindata.csv')
    
    def transform_to_window_data(self, X, y, window_size=32):
        """
        Transform the input DataFrame to a format suitable for LSTM training.
        Group up the data into windows of size. The frequent value is the label for the window.
        """
        X_new = []
        y_new = []
        count = 0
        for i in range(0, len(X) - window_size, window_size):
            X_new.append(X[i:i+window_size])
            if all(y[i:i+window_size] == 0):
                frequent_label = 0
            else:
                non_zero_labels = [value for value in y[i:i+window_size] if value != 0]
                if len(set(non_zero_labels)) == 1:
                    pass
                else:
                    count += 1
             
                activity_label = [value for value in y[i:i+window_size] if value != 0]
                # print(activity_label)
                frequent_label = scipy.stats.mode(activity_label).mode
            y_new.append(frequent_label)
        # print(f"mixed window ratio: {count/(len(X)/window_size)}")
        return X_new, y_new

if __name__ == "__main__":
    processor = DataProcessor()
    processor.extract_train_data()
    # processor.extract_test_data()
    processor.data_cleaning()
    processor.reset_label()
    processor.normalize_data()
    processor.data_collection.to_csv('OpportunityUCIDataset/dataset/s4-traindata.csv', index=False)
    # processor.data_collection.to_csv('OpportunityUCIDataset/dataset/s4-testdata.csv', index=False)