ACTION = 'test'  # 'train/test'
DATA_DIR = 'data_path'
args = {
    'BATCH_SIZE': 64,
    'EPOCH': 2,
    'HIDDEN_DIM': 300,
    'OPTIMIZER': 'Adam',  # 'Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD'
    'CRF': True,
    'LR': 0.001,
    'CLIP': 5,
    'DROPOUT': 0.5,
    'UPDATE_EMBEDDING': True,
    'PRETRAIN_EMBEDDING': 'random',
    'EMBEDDING_DIM': 300,
    'SHUFFLE': True,
    'MINI_COUNT': 1
}
