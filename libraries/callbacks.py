from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, RemoteMonitor, TerminateOnNaN, \
    BackupAndRestore
#from livelossplot import PlotLossesKeras


def get_callbacks(name="model"):
    # EarlyStopping, detener el entrenamiento una vez que su pérdida comienza a aumentar
    early_stop = EarlyStopping(
        monitor='accuracy',
        patience=8,
        # argumento de patience representa el número de épocas antes de detenerse una vez que su pérdida comienza a aumentar (deja de mejorar).
        min_delta=0,
        # es un umbral para cuantificar una pérdida en alguna época como mejora o no. Si la diferencia de pérdida es inferior a min_delta , se cuantifica como no mejora. Es mejor dejarlo como 0 ya que estamos interesados ​​en cuando la pérdida empeora.
        restore_best_weights=True,
        mode='max')

    # ReduceLROnPlateau, que si el entrenamiento no mejora tras unos epochs específicos, reduce el valor de learning rate del modelo
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=5,
        min_delta=1e-4,
        mode='min',
        verbose=1,
    )

    # Saves Keras model after each epoch
    # Para algunos casos es importante saber cual entrenamiento fue mejor,
    # este callback guarda el modelo tras cada epoca completada con el fin de si luego se desea un registro de pesos para cada epoca
    # Se ha usado este callback para poder optener el mejor modelo de pesos, sobretodo en la red neuronal creada desde cero
    # siendo de gran utilidad para determinar el como ir modificando los layer hasta obtener el mejor modelo
    checkpointer = ModelCheckpoint(
        filepath='models_backup/' + name + '-{val_accuracy:.4f}.h5',
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True,
        save_weights_only=False
    )

    remote_monitor = RemoteMonitor(
        root='http://localhost:6006',
        path='/remote_monitor/',
        field='data',
        headers=None,
        send_as_json=False
    )

    backup_restore = BackupAndRestore(backup_dir="backup")

    proteccion_nan_loss = TerminateOnNaN()

    callbacks_list = [early_stop, reduce_lr, checkpointer, proteccion_nan_loss, backup_restore, remote_monitor] # , PlotLossesKeras()] #, remote_monitor]

    return callbacks_list