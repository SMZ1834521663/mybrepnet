import os
import sys
sys.path.append('/data/smz24/mybrepnet2/')
print(os.getcwd())


import argparse
from pathlib import Path
import time

import me.my_data_utils as data_utils 

from dataloaders.max_num_faces_sampler import MaxNumFacesSampler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
#####pyOCC库绑定到原C++的OCC库：https://github.com/tpaviot/pythonocc-core
from me.config import cfg
from me.my_brepnet_gcn import Brep_Gcn
from me.my_dataset import MyBRepDataset,brepnet_collate_fn
import scipy.sparse as sp

from models.uvnet_encoders import UVNetCurveEncoder, UVNetSurfaceEncoder

import itertools

def build_adjacency(adj_dict):
    """根据邻接表创建邻接矩阵"""
    
    edge_index = []
    num_nodes = len(adj_dict)
    for src, dst in adj_dict.items():
        edge_index.extend([src, v.detach().cpu()] for v in dst)   #分别是a到b，b到a
        edge_index.extend([v.detach().cpu(), src] for v in dst)

    # 去除重复的边,没有环，但是有双边，即一个半边方向上有两个半边
    edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
    edge_index = np.asarray(edge_index)
    adjacency = sp.coo_matrix((np.ones(len(edge_index)), 
                                (edge_index[:, 0], edge_index[:, 1])),  #数据值，行索引，列索引。数据值根据行列索引xy放到shape大小的矩阵中
                shape=(num_nodes, num_nodes), dtype="float32")
    return adjacency


def normalization(adjacency):
    """计算 H=D^-0.5 * (A+I) * D^-0.5"""
    adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()#.toarray()



class MyBrepGcnTrainer():
    def __init__(self,cfg):
        self.cfg=cfg
        self.model=Brep_Gcn(self.cfg).to(cfg.device)#不要单独将数据移到cuda上，会丢失
        self.optimizer=self.get_optimizer()
        self.criterion=self.get_criterion()
        self.surface_encoder = UVNetSurfaceEncoder(output_dims=64)
        self.surface_encoder.to(cfg.device)

    def get_dataloader(self,train_val_or_test):
        dataset = MyBRepDataset(self.cfg, train_val_or_test)

        return torch.utils.data.DataLoader(
        dataset,
        batch_size=self.cfg.batch_size,
        collate_fn=brepnet_collate_fn,
        shuffle=False
        )
    
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.cfg.lr)
        return optimizer
     
    def get_criterion(self):
        criterion = nn.CrossEntropyLoss().to(self.cfg.device) #nn.CrossEntropyLoss()函数计算交叉熵损失
        return criterion

    def find_loss(self, logits, all_batch_labels):
        return F.cross_entropy(logits, all_batch_labels, reduction='mean').to(self.cfg.device)

    def find_predicted_classes(self, t):
        norm_seg_scores = F.softmax(t.detach(), dim=1)
        return torch.argmax(norm_seg_scores, dim=1)
    
    def to_cuda(self, data, device):
        
        for k in data:
            if k=="file_stems":
                continue

            if isinstance(data[k], tuple) or isinstance(data[k], list):
                data[k] = [b.to(device) for b in data[k]]
            elif isinstance(data[k], dict):
                for key,v in data[k].items():
                    data[k][key] = torch.tensor(data[k][key]).to(device)
            else:
                data[k] = data[k].to(device)
        return data

    def collate_epoch_outputs(self, outputs):
        """
        Collate information from all batches at the end of an epoch
        """
        num_faces_correct = 0
        total_num_faces = 0
        per_class_intersections = [0.0] * self.cfg.num_classes
        per_class_unions = [0.0] * self.cfg.num_classes
        for output in outputs:
            total_num_faces += output["iou_data"]["num_faces"]
            num_faces_correct += output["iou_data"]["num_faces_correct"]
            for i in range(self.cfg.num_classes):
                per_class_intersections[i] += output["iou_data"]["per_class_intersections"][i]
                per_class_unions[i] += output["iou_data"]["per_class_unions"][i]

        per_class_iou = []
        mean_iou = 0.0
        for i in range(self.cfg.num_classes):
            if per_class_unions[i] > 0.0:
                iou = per_class_intersections[i]/per_class_unions[i]
            else:
                # Should never come here with the full dataset
                iou = 1.0
            per_class_iou.append(iou)
            mean_iou += iou

        accuracy = num_faces_correct / total_num_faces
        mean_iou /= self.cfg.num_classes
        return {
            "accuracy": accuracy,
            "mean_iou": mean_iou,
            "per_class_iou": per_class_iou,
            "total_num_faces": total_num_faces
        }

    def train_once(self,data):
        
        # Unpack the tensor data
        Xf = data["face_features"]
        Xe = data["edge_features"]
        Gf = self.surface_encoder(data["face_point_grids"])
        
        labels = data["labels"]
        dual_graph=data["dual_graph"]
        edge_to_face=data["edge_to_face"]
        face_to_edge=data["face_to_edge"]
        adj_sp_matrix=build_adjacency(dual_graph)  #先试试每一次输入不同的邻接矩阵
        # norm_adj_matrix=torch.tensor(normalization(adj_sp_matrix),dtype=torch.float32).to(self.cfg.device)   # 归一化邻接矩阵,array形式

        norm_adj_sp_matrix=normalization(adj_sp_matrix)
        indices = torch.from_numpy(np.asarray([norm_adj_sp_matrix.row, norm_adj_sp_matrix.col]).astype('int64')).long()
        values = torch.from_numpy(norm_adj_sp_matrix.data.astype(np.float32))
        norm_adj_matrix = torch.sparse.FloatTensor(indices, values,(Xf.size(0), Xf.size(0))).to(self.cfg.device)  #sp的

        face_feature=Xf
        face_add_edge_feature=torch.zeros((Xf.size(0),Xe.size(1))).to(self.cfg.device)  #经过dataloader了，前面有个batchsize维度
        for key , value in edge_to_face.items():
            for k in value:
                face_add_edge_feature[k]+=Xe[key]
        feature=torch.cat((face_feature,face_add_edge_feature,Gf),axis=1)

        #添加面数特征
        face_num_feature=torch.zeros((Xf.size(0),2)).to(self.cfg.device)
        for key,value in face_to_edge.items():
            face_num_feature[key][0]=len(value)
            face_num_feature[key][1]=len(value)

        feature=torch.cat((feature,face_num_feature),axis=1)

        norm_feature = feature / feature.sum(1, keepdims=True)  # 归一化数据
        norm_feature = feature
        
        self.model.train()
        logits = self.model(norm_adj_matrix,norm_feature)
        loss = self.find_loss(logits, labels)
        
        predicted_classes = self.find_predicted_classes(logits)
        
        num_faces = labels.size(0)
        num_labels_per_face = self.cfg.num_classes  #这个东西就是8
        correct = (labels==predicted_classes)
        num_faces_correct = torch.sum(correct).item()
        accuracy = num_faces_correct/num_faces

        # Compute the per-class IoU
        per_class_intersections = [0.0] * self.cfg.num_classes
        per_class_unions = [0.0] * self.cfg.num_classes
        for i in range(num_labels_per_face):
            selected = (predicted_classes == i)
            selected_correct = (selected & correct)
            labelled = (labels == i)
            union = selected | labelled
            per_class_intersections[i] += selected_correct.sum().item()
            per_class_unions[i] += union.sum().item()

        iou_data = {
            "num_faces": num_faces,
            "num_faces_correct": num_faces_correct,
            "per_class_intersections": per_class_intersections,
            "per_class_unions": per_class_unions
        }

        return {
            "loss": loss,
            "accuracy": accuracy,
            "iou_data": iou_data
        }

    def train(self):
        print(" ")
        print("--------------------------------------------------------------------------")
        print("start training")
        print(" ")
        #获取数据集
        train_data_loader=self.get_dataloader("training_set")
        val_data_loader=self.get_dataloader("validation_set")
        #一些配置
        progress_bar = tqdm(range(0,cfg.epochs*len(train_data_loader)), desc="Training progress")
        #开始训练
        accuarcy_all=[]
        loss_all=[]
        for epoch in range(self.cfg.epochs):
            for index, data in enumerate(train_data_loader):  #一个data是一个零件的所有参数
            
                torch.cuda.empty_cache()
                data = self.to_cuda(data, self.cfg.device)
                train_pkg= self.train_once(data)
                loss=train_pkg["loss"]
                accuarcy=train_pkg["accuracy"]
                iou_data=train_pkg["iou_data"]

                cc=self.collate_epoch_outputs([train_pkg])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 

                accuarcy_all.append(accuarcy)
                loss_all.append(loss)
                progress_bar.update(1)
                if(index%100==0):
                    acc_now=sum(accuarcy_all)/len(accuarcy_all)
                    loss_now=sum(loss_all)/len(loss_all)
                    accuarcy_all=[]
                    loss_all=[]
                    progress_bar.set_postfix({'acc':acc_now,"loss":loss_now})
                    with open("/data/smz24/mybrepnet2/exp/exp.txt", 'a') as file:  # 'a' 模式用于追加
                        file.write("index:{}".format(index)+'acc:{}'.format(acc_now)+","+"loss:{}".format(loss_now) + '\n') 
            # 多少个epoch之后，全部将验证集输入并验证
            if epoch%1==0:
                print("valing")
                accuarcy_all=[]
                iou_all=[]
                for index, data in enumerate(val_data_loader):
                    data = self.to_cuda(data, self.cfg.device)
                    with torch.no_grad():  # 显著减少显存占用
                        train_pkg= self.train_once(data)
                        loss=train_pkg["loss"]
                        accuarcy=train_pkg["accuracy"]
                        iou_data=train_pkg["iou_data"]
                        accuarcy_all.append(accuarcy)
                        iou_all.append(iou_data)
                print("acc",sum(accuarcy_all)/len(accuarcy_all))
                with open("/data/smz24/mybrepnet2/exp/exp.txt", 'a') as file:  # 'a' 模式用于追加
                    file.write("test:"+'acc:{}'.format(acc_now)+","+"loss:{}".format(loss_now) + '\n') 

                



if __name__ == '__main__':

    trainer=MyBrepGcnTrainer(cfg)
    trainer.train()


    

    
