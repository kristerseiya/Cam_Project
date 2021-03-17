
from torch.utils.data import Dataset
import glob
from PIL import Image
import config

class HotDogNotHotDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.images = []
        self.labels = []

        for file_path in glob.glob(root_dir+'/hot_dog/*.jpg'):
            fptr = Image.open(file_path)
            file_copy = fptr.copy()
            fptr.close()
            self.images.append(file_copy)
            self.labels.append(1)

        for file_path in glob.glob(root_dir+'/not_hot_dog/*.jpg'):
            fptr = Image.open(file_path)
            file_copy = fptr.copy()
            fptr.close()
            self.images.append(file_copy)
            self.labels.append(0)

        if transform is None:
            self.transform = config.IMAGE_TRANFORM_TRAINING
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]
