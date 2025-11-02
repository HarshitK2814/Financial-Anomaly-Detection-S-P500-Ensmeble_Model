class TimeSeriesDataset:
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def preprocess(self):
        # Implement preprocessing steps such as normalization, scaling, etc.
        pass

    def split(self, train_size=0.8):
        split_index = int(len(self.data) * train_size)
        train_data = self.data[:split_index]
        test_data = self.data[split_index:]
        
        if self.labels is not None:
            train_labels = self.labels[:split_index]
            test_labels = self.labels[split_index:]
            return (train_data, train_labels), (test_data, test_labels)
        
        return train_data, test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index] if self.labels is not None else None


def load_dataset(file_path):
    # Implement loading logic for datasets from CSV or JSON
    pass

def save_dataset(data, file_path):
    # Implement saving logic for datasets to CSV or JSON
    pass