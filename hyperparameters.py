def load_hyperparameters():
    '''
    freeze_dense_block_num: -1 == "does not freeze", 4 == "freeze or layers except the output layer"
    '''
    return {
        'dropout_dense': 0.0,
        'weight_decay': 0.0,
        'dropout_fc': 0.0,
        'batch_size': 32,
        'freeze_dense_block_num': -1,
        'rotation_range': 20,
        'shear_range': 0.2,
        'zoom_range': 0.2
    }
