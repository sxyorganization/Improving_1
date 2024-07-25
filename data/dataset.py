import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


# 这段代码的主要目的就是从预训练的BERT模型中提取嵌入权重,并将其转换为可训练的Embedding层
# 继承自SequentialDataset
class TedRecDataset(SequentialDataset):
    def __init__(self,dataset, config,distribution='uniform'):
        self.dataset = dataset


        # 确保 dataset 是一个对象，而不是字符串
        if isinstance(dataset, str):
            raise TypeError("dataset 参数应该是一个对象，而不是字符串")
        super().__init__(config)
        ## 从配置中获取PLM的维度大小和文件后缀
        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        ## 加载PLM嵌入权重
        plm_embedding_weight = self.load_plm_embedding()
        # 将权重转换为嵌入层
        self.plm_embedding = self.weight2emb(plm_embedding_weight)
        # 初始化 head_entity_field 属性
        #self.head_entity_field = config['head_entity_field'] if 'head_entity_field' in config else None

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

    def get_embeddings_for_improving(self, user_interaction_sequence):
        # 假设 user_interaction_sequence 是用户交互的项目ID列表
        sequence_embeddings = []
        for item_id in user_interaction_sequence:
            item_embedding = self.plm_embedding(torch.tensor([item_id], dtype=torch.long))
            sequence_embeddings.append(item_embedding)
        # 将所有项目嵌入合并成一个序列嵌入
        sequence_embeddings = torch.cat(sequence_embeddings, dim=0)
        return sequence_embeddings
    #api文档新添加的内容
    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.entity_num, sample_num)
"""
    def _get_candidates_list(self):

        return list(self.hid_list) + list(self.tid_list)

    def get_used_ids(self):
        used_tail_entity_id = np.array([set() for _ in range(self.entity_num)])
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)

        for used_tail_set in used_tail_entity_id:
            if len(used_tail_set) + 1 == self.entity_num:  # [pad] is a entity.
                raise ValueError(
                    'Some head entities have relation with all entities, '
                    'which we can not sample negative entities for them.'
                )
        return used_tail_entity_id

    def sample_by_entity_ids(self, head_entity_ids, num=1):
        try:
            return self.sample_by_key_ids(head_entity_ids, num)
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError(f'head_entity_id [{head_entity_id}] not exist.')
"""

