import numpy as np
import pandas as pd
from skimage import io
import os
import json
from tqdm import tqdm


def create_list_box(img, coord, grid_box = [100, 150, 200]):
    """
    From tile and box coordinates of boat, create images with
    different size of bounding boxes (defined by grid_box)
    :param img: 3D Numpy Array (768,768,3), correspond to the tile
    :param coord: List, represents xmin, xmax, ymin, ymax of the box
    :param grid_box: List, references the number of pixel to add to the
        original bounding box
    :return: List, of 3D Numpy Array (different size)
    """
    xmin, xmax, ymin, ymax = coord
    list_box = []
    for npi in grid_box:

        ymax_npi = ymax+npi
        ymin_npi = ymin-npi
        xmax_npi = xmax+npi
        xmin_npi = xmin-npi

        if ymax_npi > 768:
            ymax_npi = 768
        if ymin_npi < 0:
            ymin_npi = 0
        if xmax_npi > 768:
            xmax_npi = 768
        if xmin_npi < 0:
            xmin_npi = 0

        list_box.append(img[ymin_npi:ymax_npi, xmin_npi:xmax_npi])
    return list_box


def save_list_box(list_box, grid_box, data_clean_dir, record_id):
    """
    Take the list_box from create_list_box and save each image (box) to
    the directory path_dir with name record_id + the number of pixel added
    :param list_box: List, of 3D Numpy Array (different size)
    :param grid_box: grid_box: List, references the number of pixel to add to the
        original bounding box
    :param data_clean_dir: String, path of the directory
    :param record_id: String, id of the boat's box
    :return: l_dir_image_clean, List, record_id and path of clean images saved
    """
    if not os.path.exists(data_clean_dir):
        os.makedirs(data_clean_dir)
    l_dir_image_clean = []
    for box, npi in zip(list_box, grid_box):
        io.imsave(os.path.join(data_clean_dir, record_id+'_'+str(npi)+".jpg"), box, check_contrast=False)
        l_dir_image_clean.append([record_id, os.path.join(data_clean_dir, record_id+'_'+str(npi)+".jpg")])
    return l_dir_image_clean


def create_list_label_sample(data_dir, dataset_name):
    """
    Generate two lists of complete path for both tiles and labels
    The two folders in dataset_name are named "samples" and "labels"
    Return intersection of the two lists by file name
    :param data_dir: String, path of the data
    :param dataset_name: name of the dataset in data_dir
    :return: list_sample_path: List, list of complete path of tiles
             list_label_path: List, list of complete path of labels
    """
    def list_sorted(name):
        list_ = []
        for dir_, _, filenames in os.walk(os.path.join(data_dir,dataset_name, name)):
            for f in filenames:
                if f[0] != ".":
                    list_.append(os.path.abspath(os.path.join(dir_, f)))
        list_.sort(key=lambda f: f.split('/')[-1])
        return list_
    list_sample_path, list_label_path = list_sorted("samples"), list_sorted("labels")
    set_sample_id = set(list(map(lambda x: x.split('/')[-1].replace(".jpg",""), list_sample_path)))
    set_label_id = set(list(map(lambda x: x.split('/')[-1].replace(".json",""), list_label_path)))
    id_diff = set_sample_id.difference(set_label_id)
    id_diff.update(set_label_id.difference(set_sample_id))
    list_sample_path = list(filter(lambda x: x.split('/')[-1].replace(".jpg","") not in id_diff, list_sample_path))
    list_label_path = list(filter(lambda x: x.split('/')[-1].replace(".json","") not in id_diff, list_label_path))
    return list_sample_path, list_label_path


def polygone_to_min_max_coordinates(polygon):
    """
    Get the min and max pixel of the polygon (casted to int)
    :param polygon: List, coordinates of the bounding box
    :return: Tuple, of int xmin, xmax, ymin, ymax
    """
    # Remove the last point that correspond to the first one
    polygon = np.array(polygon[:-1])
    # Get min and max coordinates transformed to int
    xmin = int(np.min(polygon[:, 0]))
    xmax = int(np.max(polygon[:, 0])) + 1
    ymin = int(np.min(polygon[:, 1]))
    ymax = int(np.max(polygon[:, 1])) + 1
    return xmin, xmax, ymin, ymax


def boat_info_from_json(label_path, image_path):
    """
    From a label_path and image_path get all boats info present in the image.
    Return the list of info for each boat.
    :param label_path: String, path to the label
    :param image_path: String, path to the image
    :return: l_res: List, list of basic info get from the json for each boat
    """
    # read the json file containing the boat info
    with open(label_path, 'rb') as f:
        boat_info = json.load(f)
    # l_res contains the final info to keep
    l_res = []
    for feature in boat_info["features"]:
        properties = feature["properties"]
        infos = []
        # Finding the speed tag if not exists continue
        if "tags" not in properties:
            continue
        speed = None
        for speed_tag in ["idle", "fast", "slow"]:
            if speed_tag in properties["tags"]:
                speed = speed_tag
        # If there is not a record_id or not speed tag then continue
        if "record_id" not in properties or speed is None:
            continue
        xmin, xmax, ymin, ymax = polygone_to_min_max_coordinates(feature["geometry"]["coordinates"][0])
        infos.append(properties["record_id"])
        # Appending all infos
        infos.append(xmin)
        infos.append(xmax)
        infos.append(ymin)
        infos.append(ymax)
        infos.append(properties["angle"])
        infos.append(properties["length"])
        infos.append(properties["width"])
        infos.append(properties["kept_percentage"])
        infos.append(speed)
        infos.append(image_path)
        l_res.append(infos)
    return l_res


def boat_into_to_csv(csv_name, list_label_path, list_sample_path):
    """
    Create a csv containing all information for each boat.
    The record_id duplicate are dropped by keeping the max kept_percentage
    :param csv_name: String, Name of the csv saved
    :param list_label_path: List, all label path get from create_list_label_sample
    :param list_sample_path: List, all sample path get from create_list_label_sample
    :return:
    """
    l_info = []
    col_names = ["record_id", "xmin", "xmax", "ymin", "ymax", "angle", "length", "width", "kept_percentage", "tag", "image_path"]
    for label_path, image_path in tqdm(zip(list_label_path, list_sample_path), total=len(list_label_path)):
        l_info += boat_info_from_json(label_path, image_path)
    # Transform list to Pandas dataframe
    df_info = pd.DataFrame(l_info, columns=col_names)
    # Removing duplicate record_id by keeping only the best 'kept_percentage'
    df_info = df_info.groupby('record_id', group_keys=False).apply(lambda x: x.loc[x.kept_percentage.idxmax()])
    df_info.to_csv(csv_name, sep=",", index=False)


def generate_image_and_csv_with_different_bounding_box(csv_clean_name, csv_name, data_clean_dir, grid_box):
    """
    Generate all the images with different bounding box and save the csv with information about boat
    :param csv_clean_name: String, path to the new csv file (clean)
    :param csv_name: String, path to the csv file
    :param data_clean_dir: String, path to the dir where will be stored the images
    :param grid_box: List, references the number of pixel to add to the
        original bounding box
    :return:
    """
    df_info = pd.read_csv(csv_name)
    l_dir_record_clean = []
    for index, row in tqdm(df_info.iterrows(), total=len(df_info)):
        img = io.imread(row["image_path"])
        coord = [row["xmin"], row["xmax"], row["ymin"], row["ymax"]]
        record_id = row["record_id"]
        list_box = create_list_box(img, coord, grid_box)
        l_dir_image_clean = save_list_box(list_box, grid_box, data_clean_dir, record_id)
        l_dir_record_clean += l_dir_image_clean
    df_dir_record_clean = pd.DataFrame(l_dir_record_clean, columns=["record_id", "image_clean_path"])
    df_dir_record_clean = df_dir_record_clean.set_index("record_id").join(df_info.set_index("record_id")).reset_index()
    df_dir_record_clean.to_csv(csv_clean_name, index=False)

