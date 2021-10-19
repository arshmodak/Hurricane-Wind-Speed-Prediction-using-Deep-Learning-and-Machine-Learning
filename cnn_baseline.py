import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from hurricane_dataloader import trainloader, testloader, validloader


# def get_data_loader():
#
#     data_dir = r"E:\ARSH\NEU\Fall 2021\DS 5500\Project\Data"
#     setname = "train"
#     folder_name = "nasa_tropical_storm_competition_{}_source".format(setname)
#     train_metadata = MetaData(data_dir, folder_name, setname)
#     setname = "test"
#     folder_name = "nasa_tropical_storm_competition_{}_source".format(setname)
#     test_metadata = MetaData(data_dir, folder_name, setname)
#
#     trainset = HurricaneImageDataset(train_metadata)
#     testset = HurricaneImageDataset(test_metadata)
#
#     trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
#     testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
#
#     return trainloader, testloader
#
# trainloader, testloader = get_data_loader()

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(25088, 32)
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)

        return x


# create a complete CNN
model = Net()
print(model)

# move tensors to GPU if CUDA is available
train_on_gpu = True
if train_on_gpu:
    model.cuda()

# specify loss function
criterion = nn.MSELoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.015)

# number of epochs to train the model
n_epochs = 30  # you may increase this number to train a final model

valid_loss_min = np.Inf  # track change in validation loss

losses = {"train_loss": [], "test_loss": []}

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainloader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
            target = target.float().unsqueeze(1)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # print("Size: {}".format(output.size()))
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)


    ######################
    # validate the model #
    ######################
    model.eval()
    for data, target in validloader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)


    # calculate average losses
    train_loss = train_loss / len(trainloader.dataset)
    losses["train_loss"].append(train_loss)
    valid_loss = valid_loss / len(testloader.dataset)
    losses["test_loss"].append(valid_loss)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        # torch.save(model.state_dict(), 'model_cifar_test3.pt')
        valid_loss_min = valid_loss