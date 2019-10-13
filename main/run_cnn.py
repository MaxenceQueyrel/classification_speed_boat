from model import cnn as CNN
from utils import prediction_process
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


model_name = "cnn2"
prediction_dir = "/classification_speed_boat/prediction/"
csv_preds_name = "preds.csv"
data_dir = "/classification_speed_boat/data/"
data_clean_dir = "/classification_speed_boat/data_clean/"
dataset_name = "test_students"
csv_clean_name = "info_boat.csv"
csv_info_name = "info_boat.csv"
net_name = "cnn_basic.pt"
path_model = os.path.join(prediction_dir, model_name)

input_size = 3
output_size = 3
size = 128
batch_size = 500
num_workers = 0
test_size = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn = CNN.Net(input_size, output_size)
cnn.load_state_dict(torch.load(os.path.join(path_model, net_name)))
cnn = cnn.to(device)

X = pd.read_csv(os.path.join(data_clean_dir, dataset_name, csv_clean_name))

transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
dataset = CNN.SatelliteImageDataset(X, transform, device)

loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

score, y_true, y_pred, y_softmax = CNN.compute_score(cnn, loader)
preds = np.array(list(zip(list(dataset.X["record_id"]), list(y_softmax))))

prediction_process.save_df_preds(preds, prediction_dir, csv_preds_name, model_name, dataset_name)
prediction_process.fill_data(csv_preds_name, prediction_dir, csv_info_name, model_name, data_dir, dataset_name)

