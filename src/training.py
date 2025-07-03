import os
import pandas as pd
import argparse
import logging
from src.utils.common_utils import read_yaml
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, save_plot
from src.utils.callbacks import get_callbacks


def training(config_path):

    config = read_yaml(config_path)
    logging.info(">>> Data Loading and splitting started>>>>")
    validation_datasize = config['params']['VALIDATION_DATASIZE']
    (X_train, y_train),(X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    LOSS_FUNCTION = config['params']['LOSS_FUNCTION']
    OPTIMIZER = config['params']['OPTIMIZER']
    METRICS = config['params']['METRICS']
    NUM_CLASSES = config['params']['NUM_CLASSES']
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    EPOCHS = config['params']['EPOCHS']
    VALIDATION  = (X_valid, y_valid)
    logging.info(">>> Data Loading and splitting completed>>>>")

    
    CALLBACK_LIST = get_callbacks(config, X_train)

    history = model.fit(X_train, y_train, epochs = EPOCHS, validation_data = VALIDATION, callbacks = CALLBACK_LIST)
    model_name = config ['artifacts']['MODEL_NAME']
    model_dir = config['artifacts']['MODEL_DIR']
    os.makedirs(model_dir, exist_ok=True) #directory is the folder name, not the path
    model_dir_path = os.path.join("artifacts", model_dir) # pathincludes filename
    #os.makedirs(model_dir_path, exist_ok=True)
    save_model(model, model_name, model_dir_path)


    plt = pd.DataFrame(history.history).plot(figsize = (10, 7))
    plt_name = config['artifacts']['plots_name']
    plt_dir = config['artifacts']['PLOTS_DIR']
    os.makedirs(plt_dir, exist_ok=True)
    plt_dir_path = os.path.join("artifacts", plt_dir)
    #os.makedirs(plt_dir_path, exist_ok=True)
    save_plot(plt, plt_name, plt_dir_path )
    





if __name__ == '__main__':

    #logging
    logging_str = "[%(asctime)s :  %(levelname)s : %(module)s] : %(message)s"
    log_dir = "logs/general_logs"
    os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
    logging.basicConfig(filename=os.path.join(log_dir, 'running_log.log'),
    filemode='a',  # 'w' to overwrite the log file, 'a' to append
    format=logging_str,
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(">>> General logging started >>>>")

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)

