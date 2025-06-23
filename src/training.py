import argparse
from src.utils.common_utils import read_yaml
from src.utils.data_mgmt import get_data


def training(config_path):
    config = read_yaml(config_path)
    #print(config)
    validation_datasize = config['params']['VALIDATION_DATASIZE']
    print(f" {validation_datasize}")
    (X_train, y_train),(X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)

