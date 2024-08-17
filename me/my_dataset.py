import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

import me.my_data_utils as data_utils

class MyBRepDataset(Dataset):
    def __init__(self, cfg, train_val_or_test):  #可以创建不同的数据集
        super(MyBRepDataset, self).__init__()
        self.cfg = cfg
        self.feature_lists = data_utils.load_json_data(self.cfg.input_features)
        self.dataset_dir = Path(self.cfg.dataset_dir)   #处理过的processed文件夹
        self.dataset_file=self.dataset_dir / "dataset.json"
        self.seg_dir=self.dataset_dir.parent / "breps/seg"
        dataset_info = data_utils.load_json_data(self.dataset_file)  #包含训练集，验证集，测试集，归一化参数
        self.body_file_stems = dataset_info[train_val_or_test]
        self.feature_standardization = dataset_info["feature_standardization"]

    def __len__(self):
        return len(self.body_file_stems)

    def __getitem__(self, idx):   #获取数据
        assert idx < len(self.body_file_stems)
        body_data = self.load_body(idx)
        return body_data

    def load_labels(self, file_stem):
        label_pathname = self.seg_dir / (file_stem + ".seg")
        face_labels = np.loadtxt(label_pathname, dtype=np.int64)
        face_labels_tensor = torch.from_numpy(face_labels)
        if face_labels_tensor.ndim == 0:
            face_labels_tensor = torch.unsqueeze(face_labels_tensor, 0)
        return face_labels_tensor

    def load_body(self, idx):
        file_stem = self.body_file_stems[idx]
        npz_pathname = self.dataset_dir / (file_stem + ".npz")
        body_data = data_utils.load_npz_data(npz_pathname)
        Xf, Xe, Xc = self.build_input_feature_tensors(body_data)   #归一化了
        dual_graph,edge_to_face=body_data["dual_graph"][()],body_data["edge_to_face"][()] #先试试
        Gf=body_data["face_point_grids"].astype(np.float32)
        labels = self.load_labels(file_stem)
        

        data = {
            "face_features": Xf,   
            "edge_features": Xe,
            "coedge_features": Xc,
            "face_point_grids":Gf,
            "dual_graph":dual_graph,
            "edge_to_face":edge_to_face,
            "labels": labels,
            "file_stem": file_stem
        }
        return data

    def build_input_feature_tensors(self, body_data):
        """
        Convert the feature tensors for faces, edges and coedges
        from numpy to pytorch
        """
        Xf = torch.from_numpy(body_data["face_features"])  #12*7
        Xe = torch.from_numpy(body_data["edge_features"])
        Xc = torch.from_numpy(body_data["coedge_features"])

        Xf = self.standardize_features(
            Xf, 
            self.feature_standardization["face_features"]
        )
        Xe = self.standardize_features(
            Xe, 
            self.feature_standardization["edge_features"]
        )
        Xc = self.standardize_features(
            Xc, 
            self.feature_standardization["coedge_features"] 
        )

        return Xf, Xe, Xc


    def standardize_features(self, feature_tensor, stats):
        num_features = len(stats)
        assert feature_tensor.size(1) == num_features
        means = torch.zeros(num_features, dtype=feature_tensor.dtype)
        sds = torch.zeros(num_features, dtype=feature_tensor.dtype)
        eps = 1e-7
        for index, s in enumerate(stats):
            assert s["standard_deviation"] > eps, "Feature has zero standard deviation"
            means[index] = s["mean"]
            sds[index] = s["standard_deviation"]

        # We need to broadcast means and sds over the number of entities
        means.unsqueeze(0)
        sds.unsqueeze(0)
        feature_tensor_zero_mean = feature_tensor - means
        feature_tensor_standadized = feature_tensor_zero_mean / sds

        # Test code to check this works
        num_ents = feature_tensor.size(0) 
        test_tensor = torch.zeros((num_ents, num_features), dtype=feature_tensor.dtype)
        for i in range(num_ents):
            for j in range(num_features):
                value = (feature_tensor[i,j] - means[j])/sds[j]
                test_tensor[i,j] = value
        assert torch.allclose(feature_tensor_standadized, test_tensor, eps)

        return feature_tensor_standadized.float()


    



def get_face_to_edge(edge_to_face,num_faces):
    face_to_edge=dict([])
    for i in range(num_faces):
        face_to_edge[i]=[]
    for key,value in edge_to_face.items():
        for v in value:
            face_to_edge[v].append(key)
    return face_to_edge



#批处理，是放在torch.utils.data.DataLoader中的，设置了batch之后先自动将batch的东西输入到这里，然后返回值再输入到网络中
#可以先不用，没有硬性要求
#我们自己对batch的数据处理有个好处，因为自动生成的前面会多一个batch维，自己的函数可以直接concat掉
def brepnet_collate_fn(data_list):
    Xe_list = []
    Xf_list = []
    Gf_list = []
    labels_list = []
    file_stems = []
    face_split_batch = []
    edge_split_batch = []

    dual_graph_batch=dict([])
    edge_to_face_batch=dict([])
    face_to_edge_batch=dict([])

    face_offset = 0
    edge_offset = 0


    for data in data_list:   #200个算一个batch，这是可以配置的
        num_faces = data["face_features"].shape[0]
        num_edges = data["edge_features"].shape[0]

        Xe = data["edge_features"]
        Xf = data["face_features"]    #面特征，7大类，是拓扑学的分类结果，onehot形式
        Gf = torch.tensor(data["face_point_grids"])  
        Xe_list.append(Xe)
        Xf_list.append(Xf)      
        Gf_list.append(Gf)

        labels = data["labels"]
        labels_list.append(labels)

        stem = data["file_stem"]
        file_stems.append(stem)
 
        
        face_split=torch.arange(face_offset, face_offset+num_faces, dtype=torch.int64)
        edge_split=torch.arange(edge_offset, edge_offset+num_edges, dtype=torch.int64)
        face_split_batch.append(face_split)
        edge_split_batch.append(edge_split)

        ###################
        dual_graph=data["dual_graph"]
        edge_to_face=data["edge_to_face"]

        face_to_edge=get_face_to_edge(edge_to_face,num_faces)

        dual_graph = {key + face_offset: list(np.array(value)+face_offset) for key, value in dual_graph.items()}
        dual_graph_batch.update(dual_graph)
        edge_to_face = {key + edge_offset: list(np.array(value)+face_offset) for key, value in edge_to_face.items()}
        edge_to_face_batch.update(edge_to_face)

        face_to_edge = {key + face_offset: list(np.array(value)+edge_offset) for key, value in face_to_edge.items()}
        face_to_edge_batch.update(face_to_edge)
        ###################

        face_offset += num_faces
        edge_offset += num_edges

    batch_data = {
        "face_features": torch.cat(Xf_list), #n,7     #bigfaces是11个，n=2451
        "face_point_grids": torch.cat(Gf_list),#n,7,10,10，n=2451
        "edge_features": torch.cat(Xe_list),#n,10，n=5667
        "dual_graph":dual_graph_batch,
        "edge_to_face":edge_to_face_batch,
        "labels": torch.cat(labels_list),#n，和面特征一样的n 2451
        "face_split_batch": face_split_batch,
        "edge_split_batch": edge_split_batch,
        "face_to_edge":face_to_edge_batch,
        "file_stems": file_stems#200
    }
    return batch_data