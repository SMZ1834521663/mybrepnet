from lib.datasets.gaussian_dataset_acc_merge import loadZjuMocap

def load_data(data_name, dataset):
    if data_name == 'zju_mocap':
        return loadZjuMocap(dataset)
    