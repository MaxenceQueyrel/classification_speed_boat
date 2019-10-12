from model import baseline
from utils import prediction_process

model_name = "baseline"
prediction_dir = "/classification_speed_boat/prediction/"
csv_preds_name = "preds.csv"
data_dir = "/classification_speed_boat/data/"
data_clean_dir = "/classification_speed_boat/data_clean/"
dataset_name = "train"
csv_name = "info_boat.csv"
csv_clean_name = "info_boat.csv"

preds = baseline.algo_baseline(csv_clean_name, data_clean_dir, dataset_name)
prediction_process.save_df_preds(preds, prediction_dir, csv_preds_name, model_name, dataset_name)
prediction_process.fill_data(csv_preds_name, prediction_dir, csv_name, model_name, data_dir, dataset_name)

