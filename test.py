import pandas as pd
import torch
import numpy as np
from hurricane_dataloader import trainloader,testloader, validloader, valid_metadata, test_metadata, train_metadata
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cnn_baseline import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def plot_predictions(wind_speed,metadata):
    sample_img = metadata[metadata.wind_speed == wind_speed][["image_path","predictions"]].iloc[:5]
    for i,img in enumerate(sample_img.image_path):
        # im = Image.open(img) 
        # im.show(title = sample_img["predictions"].iloc[i])
        image = mpimg.imread(img)
        plt.title("Actual: "+ str(wind_speed) +"   Predicted: " + str(round(sample_img["predictions"].iloc[i],2)))
        plt.imshow(image, cmap = "gray")
        plt.show()
    return 

def loadModel(model_path ,model = None):
  state_dict = torch.load(model_path)
  model.load_state_dict(state_dict)
  return model

def model_evaluation(dataloader,model):
    # specify loss function
    predictions = torch.FloatTensor().cuda()
    criterion = nn.MSELoss()
    total_MSE_loss, total_RMSE_loss = 0,0
    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.015)
    model.eval()
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        target = target.float().unsqueeze(1)
        output = model(data)
        predictions = torch.cat((predictions, output), 0)
        #RMSE
        MSE_loss = criterion(output, target)
        RMSE_loss = torch.sqrt(MSE_loss)
        total_MSE_loss += MSE_loss.item() * data.size(0)
        total_RMSE_loss += RMSE_loss.item() * data.size(0)
    total_MSE_loss = total_MSE_loss / len(dataloader.dataset)
    total_RMSE_loss = total_RMSE_loss / len(dataloader.dataset)
  # pass entire testloader, get outputs and append to test_metadata
    px = pd.DataFrame(predictions.cpu(), columns = ['predictions']).astype("float")
    return px, total_MSE_loss, total_RMSE_loss

model = Net()
model_path = r"D:\Northeastern courses\DS 5500\project\baselineCNN_best_model.pt"
model = loadModel(model_path,model)
model.cuda()
predictions,total_MSE_loss,total_RMSE_loss = model_evaluation(validloader,model)
res_df = pd.concat([valid_metadata, predictions], axis=1)
plot_predictions(valid_metadata.wind_speed[0],res_df)
print(total_MSE_loss,total_RMSE_loss)