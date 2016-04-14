import os, json

base_dir = "/home/cs5604s16_cl/Spring2016_IR_Project/data"
data_dir = os.path.join(base_dir , "small_data")
models_dir = os.path.join(base_dir, "models")
predictions_dir = os.path.join(base_dir, "predictions")
FP_dir = os.path.join(base_dir, "FPGrowth")
config_file = os.path.join(base_dir , "collections_config.json")

def load_config(config_file):
    """
    Load collection configuration file.
    """
    with open(config_file) as data_file:
        config_data = json.load(data_file)
    return config_data