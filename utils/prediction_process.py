import pandas as pd
from tqdm import tqdm
import os
import subprocess
import json
import numpy as np


def save_df_preds(preds, prediction_dir, csv_preds_name, model_name, dataset_name):
    """
    Take the output of a model, apply a vote between each prediction for the same boat
     and save a csv file with all tag results
    :param preds: 2D Numpy Array, first column il the record_id, second columns represents
        the prediction as an array of three probabilities with order [idle, slow, fast]
    :param prediction_dir: String, path where are stored the outputs of the predictions
    :param csv_preds_name: String, name of the csv file that contains predictions
    :param model_name: String, name of the model that computed the predictions
    :param dataset_name: String, name of the dataset used to compute prediction
    :return:
    """
    dict_class = {0: "idle", 1: "slow", 2: "fast"}
    df_preds = pd.DataFrame(preds, columns=["record_id", "preds"])
    df_preds = df_preds.groupby("record_id")["preds"].apply(np.mean).reset_index()
    df_preds["preds_tag"] = df_preds["preds"].apply(lambda x: dict_class[x.argmax()])
    model_dir = os.path.join(prediction_dir, model_name, dataset_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    df_preds.to_csv(os.path.join(model_dir, csv_preds_name))


def fill_data(csv_preds_name, prediction_dir, csv_info_name, model_name, data_dir, dataset_name):
    """

    :param csv_preds_name: String, name of the csv file that contains predictions
    :param prediction_dir: String, path where are stored the outputs of the predictions
    :param csv_info_name: String, name of the csv contains boat info
    :param model_name: String, name of the model that computed the predictions
    :param data_dir: String, path of the data
    :param dataset_name: String, name of the dataset in data_dir
    :return:
    """
    json_labels_dir_prediction = os.path.join(prediction_dir, model_name, dataset_name, "labels")
    json_labels_dir = os.path.join(data_dir, dataset_name, "labels")
    csv_preds_name = os.path.join(prediction_dir, model_name, dataset_name, csv_preds_name)
    csv_info_name = os.path.join(data_dir, dataset_name, csv_info_name)

    if os.path.exists(json_labels_dir_prediction):
        subprocess.call(["rm", "-r", json_labels_dir_prediction])
    subprocess.call(["cp", "-r", json_labels_dir, json_labels_dir_prediction])

    df_preds = pd.read_csv(csv_preds_name)
    dict_record_tag = dict(df_preds[["record_id", "preds_tag"]].values)
    df_info = pd.read_csv(csv_info_name)
    df = df_preds.set_index("record_id").join(df_info.set_index("record_id")).reset_index()
    df = df.groupby("label_path")["record_id"].agg(list).reset_index()
    df["label_path"] = df["label_path"].apply(lambda x: x.replace(json_labels_dir, json_labels_dir_prediction))

    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        with open(rows["label_path"], 'rb') as f:
            json_curr = json.load(f)
        for i, feature in enumerate(json_curr["features"]):
            try:
                json_curr["features"][i]["properties"]["tags"] = [dict_record_tag[feature["properties"]["record_id"]]]
            except:
                continue
        with open(rows["label_path"], 'w') as f:
            json.dump(json_curr, f)

