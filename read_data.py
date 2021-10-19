import pandas as pd
import os
import json
import pickle

def read_files(n1):
    label_path = "dataset/nasa_tropical_storm_competition_{}_labels/".format(n1)
    source_path = "dataset/nasa_tropical_storm_competition_{}_source/".format(n1)
    data = list()
    for label_folder,source_folder in zip(os.listdir(label_path)[1:],os.listdir(source_path)[1:]):
        dict_ = dict()
        name_ = label_folder[-7:]
        dict_["image_id"] = name_
        with open(label_path+label_folder+"/"+os.listdir(label_path+label_folder)[0],'rb') as fp:
            dict_.update(json.load(fp))
        with open(source_path+source_folder+"/"+os.listdir(source_path+source_folder)[0],'rb') as fp:
            dict_.update(json.load(fp))
        dict_['image_path'] = source_path+source_folder+"/"+os.listdir(source_path+source_folder)[1]
        data.append(dict_)
        
    return pd.DataFrame.from_dict(data, orient='columns')

train = read_files(n1= "train")
test =  read_files(n1= "test")

train.to_csv(r"outputs\train.csv", index = False)
test.to_csv(r"outputs\test.csv", index = False)
print("test")