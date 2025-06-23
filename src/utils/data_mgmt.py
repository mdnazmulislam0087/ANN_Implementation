import tensorflow as tf

def get_data(val_size):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test)  =mnist.load_data()
    X_valid, X_train = X_train_full[:val_size]/255.0, X_train_full[val_size:]/255.0
    y_valid, y_train = y_train_full[:val_size], y_train_full[val_size:]
    X_test = X_test / 255.0
    return (X_train, y_train),(X_valid, y_valid), (X_test, y_test)