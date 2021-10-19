import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import torch
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)

#------------------ARSH-------------------
data_dir = r"E:\ARSH\NEU\Fall 2021\DS 5500\Project\Data"
train_metadata = pd.read_csv(r"{}\train.csv".format(data_dir))
test_metadata = pd.read_csv(r"{}\test.csv".format(data_dir))
msk = np.random.rand(len(train_metadata)) < 0.8
train_metadata2 = train_metadata[msk]
valid_metadata = train_metadata[~msk]

train_folder_name = "nasa_tropical_storm_competition_train_source"
test_folder_name = "nasa_tropical_storm_competition_test_source"
train_image_dir = r"{}\{}".format(data_dir, train_folder_name)
test_image_dir = r"{}\{}".format(data_dir, test_folder_name)

def get_image_paths(data, image_dir, folder_name):
    data["image_path"] = image_dir + "\\" + folder_name + "_" + data["image_id"] + "\\" + "image.jpg"
    data = data[data.columns[[0, 2, 5, 1, 3, 4]]]
    return data

train_metadata = get_image_paths(train_metadata, train_image_dir, train_folder_name)
test_metadata = get_image_paths(test_metadata, test_image_dir, test_folder_name)
valid_metadata = get_image_paths(valid_metadata, train_image_dir, train_folder_name)

class HurricaneImageDataset(Dataset):

    def __init__(self, metadata, transforms):
        self.metadata = metadata
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.metadata["image_path"][index]
        # hurricane_image = io.imread(image_path)
        hurricane_image = Image.open(image_path)
        label = self.metadata["wind_speed"][index]

        if self.transforms:
            hurricane_image = self.transforms(hurricane_image)

        return hurricane_image, label

transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=1),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
])

trainset = HurricaneImageDataset(train_metadata, transform)
validset = HurricaneImageDataset(valid_metadata, transform)
testset = HurricaneImageDataset(test_metadata, transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
validloader = DataLoader(validset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=True)



#-----------------OMKAR-------------------------
# train_metadata = pd.read_csv(r"outputs\train.csv")
# test_metadata = pd.read_csv(r"outputs\test.csv")
#----------------------------------------------

# class MetaData():
#     def __init__(self, data_dir, folder_name, setname):
#         self.data_dir = data_dir
#         self.folder_name = folder_name
#         self.setname = setname
#         self.data = pd.read_csv(r"{}\{}.csv".format(self.data_dir, self.setname))
#         self.image_dir = r"{}\{}".format(self.data_dir, self.folder_name)
#         self.metadata = self.get_image_paths()
#
#     def __len__(self):
#         return len(self.data)
#
#     def get_image_paths(self):
#         self.data["image_path"] = self.image_dir + "\\" + self.folder_name + "_" + self.data["image_id"] + "\\" + "image.jpg"
#         self.data = self.data[self.data.columns[[0, 2, 5, 1, 3, 4]]]
#
#         return self.data

# class HurricaneImageDataset(Dataset):
#     def __init__(self, metadata):
#         self.metadata = metadata
#         self.transforms = transforms.Compose([
#             # transforms.RandomResizedCrop(224),
#             transforms.Resize((224, 224)),
#             transforms.Grayscale(num_output_channels=1),
#             # transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             # transforms.Normalize([0.5], [0.5])
#         ])
#
#     def __len__(self):
#         return len(self.metadata)
#
#     def get_image_paths(self, data, image_dir, folder_name):
#         data["image_path"] = image_dir + "\\" + folder_name + "_" + data["image_id"] + "\\" + "image.jpg"
#         data = data[data.columns[[0, 2, 5, 1, 3, 4]]]
#         return data
#
#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#
#         image_path = self.metadata["image_path"][index]
#         # hurricane_image = io.imread(image_path)
#         hurricane_image = Image.open(image_path)
#         label = self.metadata["wind_speed"][index]
#
#         if self.transforms:
#             hurricane_image = self.transforms(hurricane_image)
#
#         return hurricane_image, label
#
#
# trainset = HurricaneImageDataset(train_metadata, transform)
# testset = HurricaneImageDataset(test_metadata, transform)
#
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
# testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=0)




# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# images = images.numpy()
# plt.imshow(np.transpose(images[0]), cmap = "gray")
# plt.show()
# testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=0)


print(1)

