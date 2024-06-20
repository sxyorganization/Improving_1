import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset

#这段代码的主要目的就是从预训练的BERT模型中提取嵌入权重,并将其转换为可训练的Embedding层
#继承自SequentialDataset
class TedRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        ## 从配置中获取PLM的维度大小和文件后缀
        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        ## 加载PLM嵌入权重
        plm_embedding_weight = self.load_plm_embedding()
        # 将权重转换为嵌入层
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    # 加载PLM嵌入权重的函数
    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        # 从文件中加载特征，按照PLM的大小进行重塑
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        # 初始化一个全零的映射特征矩阵
        mapped_feat = np.zeros((self.item_num, self.plm_size))
        # 遍历item_id字段的每个token
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            ## 如果token是填充符，则跳过
            mapped_feat[i] = loaded_feat[int(token)]
        ## 返回映射后的特征
        return mapped_feat

    # 将权重转换为嵌入层的函数
    def weight2emb(self, weight):
        # 创建一个嵌入层，大小为item数量和PLM维度
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        ## 设置嵌入层的权重不需要梯度（不参与训练）
        plm_embedding.weight.requires_grad = True
        ## 将传入的权重复制到嵌入层的权重中
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding
