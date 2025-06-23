import argparse
from src.utils.common_utils import read_yaml
from src.utils.data_mgmt import get_data
from src.utils.model import create_model


def training(config_path):
    config = read_yaml(config_path)
    #print(config)
    validation_datasize = config['params']['VALIDATION_DATASIZE']
    print(f" {validation_datasize}")
    (X_train, y_train),(X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    LOSS_FUNCTION = config['params']['LOSS_FUNCTION']
    OPTIMIZER = config['params']['OPTIMIZER']
    METRICS = config['params']['METRICS']
    NUM_CLASSES = config['params']['NUM_CLASSES']
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    EPOCHS = config['params']['EPOCHS']
    VALIDATION  = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs = EPOCHS, validation_data = VALIDATION)
    





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)

