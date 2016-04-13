base_dir = "~/Spring2016_IR_Project/data/"
data_dir = os.path.join(base_dir , "small_data")
models_dir = os.path.join(data_dir, "models")
predictions_dir = os.path.join(data_dir, "predictions")
FP_dir = os.path.join(data_dir, "FPGrowth")
config_file = "collections_config.json"
config_data = load_config(os.path.join(base_dir , config_file))
