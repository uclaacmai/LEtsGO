import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import shutil


class LegoDataset(Dataset):
    """Custom dataset class for Lego Minifigure dataset."""

    def __init__(self, img_dir='data/', test=False, transform=None, target_transform=None):
        """
        initialize a LegoDataset instance.
        Images are colored, and of shape (3, 512, 512)

        Keyword arguments:
        img_dir -- the path to the root image directory of test and train data (default: 'data/')
        test -- True to load test data (default: False)
        transform -- transform to apply to X (default: None)
        target_transform -- transform to apply to y (default: None)
        """

        self.transform = transform
        self.target_transform = target_transform
        self.test = test
        self.img_dir = img_dir
        self.specific_dir = os.path.join(img_dir, 'test/') if test else os.path.join(img_dir, 'train/')
        self.adjusted_dir = img_dir if test else self.specific_dir
        self.full_df = None

        # if data not present, then download and format it
        if not set(['test', 'train']).issubset(os.listdir(img_dir)):
            # use kaggle api to download dataset
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files('ihelon/lego-minifigures-classification', path=img_dir, unzip=True)

            # rearrange the files
            shutil.move(os.path.join(img_dir, 'test.csv'), os.path.join(img_dir, 'test/', 'test.csv'))
            shutil.copy(os.path.join(img_dir, 'metadata.csv'), os.path.join(img_dir, 'test/', 'metadata.csv'))

            # training files
            os.mkdir(os.path.join(img_dir, 'train'))
            to_train = ['harry-potter',
                        'jurassic-world',
                        'star-wars',
                        'marvel',
                        'index.csv',
                        'metadata.csv']
            for f in to_train:
                shutil.move(os.path.join(img_dir, f), os.path.join(img_dir, 'train/', f))

        # read path names and class names from csv files
        meta_df = pd.read_csv(os.path.join(self.specific_dir, "metadata.csv"))
        if self.test:
            test_df = pd.read_csv(os.path.join(self.specific_dir, "test.csv"))
            self.full_df = test_df.merge(meta_df, on="class_id")
        else:
            train_df = pd.read_csv(os.path.join(self.specific_dir, "index.csv"))
            self.full_df = train_df.merge(meta_df, on="class_id")

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        row = self.full_df.iloc[idx]
        image = read_image(os.path.join(self.adjusted_dir, row["path"])).float()
        label = row["class_id"] - 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label