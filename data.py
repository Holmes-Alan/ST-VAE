from os.path import join
from torchvision import transforms
from datasets import DatasetFromFolder, DatasetFromVideo
from torch.utils.data import DataLoader


def transform():
    return transforms.Compose([
        # ColorJitter(hue=0.3, brightness=0.3, saturation=0.3),
        # RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])



def get_training_set(data_dir):
    content_dir = data_dir + '/content'
    ref_dir = data_dir + '/style'
    train_set = DatasetFromFolder(data_dir, ref_dir)

    # Pytorch train and test sets
    # tensor_dataset = torch.utils.data.TensorDataset(train_set)

    return train_set





def get_testing_set(test_dir, data_augmentation):

    test_set = DatasetFromFolder(test_dir, data_augmentation, transform=transform())

    # Pytorch train and test sets
    # tensor_dataset = torch.utils.data.TensorDataset(train_set)

    return test_set


