
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from pathlib import Path

import utils.data_utils as data_utils


from config import cfg


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() #初始化w
 
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight) 
        if self.use_bias:
            init.zeros_(self.bias)
            
 
    def forward(self, adjacency, input_feature):#adjacency: torch.sparse.FloatTensor
        support = torch.mm(input_feature, self.weight)
        # output = torch.mm(adjacency, support) 
        output = torch.sparse.mm(adjacency, support)  #稀疏矩阵计算，优化内存，左乘邻接矩阵代表筛选关系
        if self.use_bias:
            output += self.bias
        return output



class Brep_Gcn(nn.Module):
    def __init__(self, cfg,input_dim=83):
        super(Brep_Gcn, self).__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(in_channels=1, out_channels=4 ,kernel_size=5,padding=2)
        self.relu=nn.ReLU()
        self.gcn1 = GCN(input_dim, 1024)
        self.gcn2 = GCN(1024, self.cfg.num_classes)
        
        input_feature_metadata = data_utils.load_json_data(self.cfg.input_features) #all.json

        segment_names_file_path = Path(self.cfg.dataset_dir).parent / "segment_names.json"   #这个其实就是八大类
        self.segment_names = data_utils.load_json_data(segment_names_file_path)

    def forward(self, adjacency, feature):
        x=self.conv(feature.unsqueeze(1))
        x=torch.sum(x,dim=1)
        x=self.relu(x)
        h = F.relu(self.gcn1(adjacency, x))
        logits = self.gcn2(adjacency, h)
        return logits



    