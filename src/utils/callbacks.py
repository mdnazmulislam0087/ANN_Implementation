import tensorflow as tf
import os 
import numpy as np
import time


def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "-")
    unique_dir_name = f"{name}_{timestamp}" 
    return unique_dir_name

def get_callbacks(config, X_train):
    logs = config['logs']
    unique_dir_name = get_timestamp("tb_logs")
    TENSORBOARD_LOG_DIR = os.path.join(logs['LOGS_DIR'], logs['TENSORBOARD_ROOT_LOG_DIR'], unique_dir_name)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = TENSORBOARD_LOG_DIR)
    file_writer = tf.summary.create_file_writer(TENSORBOARD_LOG_DIR)
    with file_writer.as_default():
        images = np.reshape(X_train[:25], (-1, 28, 28, 1))
        tf.summary.image("Training data", images, max_outputs=25, step=0)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = config['params']['patience'], restore_best_weights=config['params']['restore_best_weights'])
    
    check_point_name = "model_ckpt.keras"
    model_checkpoint_dir = os.path.join(config['artifacts']['ARTIFACTS_DIR'], config['artifacts']['CHECKPOINTS_DIR'])
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(model_checkpoint_dir, check_point_name)
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath = model_checkpoint_path,
        save_best_only = True,
    )

    return [tensorboard_callback, early_stopping_cb, model_checkpoint_cb]


