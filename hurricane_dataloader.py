import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io

pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
data_dir = r"E:\ARSH\NEU\Fall 2021\DS 5500\Project\Data"
train_metadata = pd.read_csv(r"{}\train.csv".format(data_dir))
test_metadata = pd.read_csv(r"{}\test.csv".format(data_dir))

train_folder_name = "nasa_tropical_storm_competition_train_source"
test_folder_name = "nasa_tropical_storm_competition_test_source"
train_image_dir = r"{}\{}".format(data_dir, train_folder_name)
test_image_dir = r"{}\{}".format(data_dir, test_folder_name)

def get_image_paths(data, image_dir, folder_name):
    data["image_path"] = image_dir + "\\" + folder_name + "_" + data["image_id"] + "\\" + "image.jpg"
    data = data[data.columns[[0, 2, 5, 1, 3, 4]]]
    return data

train_metadata = get_image_paths(train_metadata, train_image_dir, train_folder_name)
test_metadata = get_image_paths(test_metadata, test_image_dir, train_folder_name)


class HurricaneImageDataset(Dataset):

    def __init__(self, metadata, transforms):
        self.metadata = metadata
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = metadata["image_path"][index]
        hurricane_image = io.imread(image_path)
        label = metadata["wind_speed"][index]

        if self.transforms:
            hurricane_image, label = self.transform(hurricane_image), self.transforms(label)

        return hurricane_image, label

transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = HurricaneImageDataset(train_metadata, transform)
testset = HurricaneImageDataset(test_metadata, transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_worker=0)
testloader = DataLoader(testset, batch_size=32, shuffle=True, num_worker=0)





#
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

print(1)

