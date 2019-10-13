from utils import data_cleaning

# Parameters
data_dir = "/classification_speed_boat/data/"
data_clean_dir = "/classification_speed_boat/data_clean/"
dataset_name = "test_students"
csv_info_name = "info_boat.csv"
csv_clean_name = "info_boat.csv"
grid_box = [100, 150, 200]
test_mode = True

# Run
list_sample_path, list_label_path = data_cleaning.create_list_label_sample(data_dir, dataset_name)
data_cleaning.boat_info_to_csv(csv_info_name, data_dir, dataset_name, list_label_path, list_sample_path, test_mode)
data_cleaning.generate_image_and_csv_with_different_bounding_box(csv_clean_name, csv_info_name, data_clean_dir,
                                                                 data_dir, dataset_name, grid_box)
