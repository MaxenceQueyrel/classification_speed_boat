from utils import data_cleaning

# Parameters
data_dir = "/classification_speed_boat/data/"
data_clean_dir = "/classification_speed_boat/data_clean/images"
dataset_name = "train"
csv_name = "/classification_speed_boat/data/info_boat.csv"
csv_clean_name = "/classification_speed_boat/data_clean/info_boat.csv"
grid_box = [100, 150, 200]

# Run
list_sample_path, list_label_path = data_cleaning.create_list_label_sample(data_dir, dataset_name)
data_cleaning.boat_into_to_csv(csv_name, list_label_path, list_sample_path)
data_cleaning.generate_image_and_csv_with_different_bounding_box(csv_clean_name, csv_name, data_clean_dir, grid_box)
