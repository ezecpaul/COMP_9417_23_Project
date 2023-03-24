import os
class paths:
    ROOT_DIR = os.path.dirname(os.path.abspath('23_9417_Project'))
    DATA_DIR = os.path.join(paths.ROOT_DIR, 'data')
    DATA_PKL = os.path.join(paths.ROOT_DIR, 'data_pkl')
    DATA_PRO = os.path.join(paths.ROOT_DIR, 'data_processing')
    MODEL_PKL = os.path.join(paths.ROOT_DIR, 'model_pkl')
    MODEL_DIR = os.path.join(paths.ROOT_DIR, 'models')