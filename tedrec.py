import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.knowledge_aware_recommender.kgcn import KGCN
from recbole.model.layers import TransformerEncoder, VanillaAttention
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType
import networkx as nx
import numpy as np

    
class DTRLayer(nn.Module):
    """Distinguishable Textual Representations Layer
    为了增强BERT语义特征的可区分性,使其能够与项目ID信息更好地融合,从而提升推荐系统的性能
    """

    def __init__(self, input_size, output_size, dropout=0.0, max_seq_length=50):
        super(DTRLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(1, max_seq_length, input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module): 
    """MoE-enhanced Adaptor
    MoEAdaptorLayer进一步引入了Mixture of Experts的自适应机制,让模型能够根据不同的输入,自动选择和组合最合适的特征变换
    """
    def __init__(self, n_exps, layers, dropout=0.0, max_seq_length=50, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([DTRLayer(layers[0], layers[1], dropout, max_seq_length) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class TedRec(SASRec):
    #Text-ID fusion approach for sequential recommendation

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        #这里设置了一个名为 temperature 的属性，它的值从传入的 config 字典中获取。
        # 在深度学习中，温度参数通常用于控制输出分布的平滑程度，尤其是在softmax函数中。
        self.temperature = config['temperature']
        #plm_embedding 可能代表预训练语言模型（Pre-trained Language Model）的嵌入层。
        #这两行代码创建了两个线性层（nn.Linear），它们通常用于神经网络中的线性变换。
        # 这里它们被用作门控机制（gating mechanism），可能用于控制信息流。
        self.item_gating = nn.Linear(self.hidden_size, 1)
        self.fusion_gating = nn.Linear(self.hidden_size, 1)
        #初始化了一个名为 MoEAdaptorLayer 的层，这可能是一个“专家混合”（Mixture of Experts）适配器层
        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob'],
            self.max_seq_length
        )
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        #创建了一个复数权重矩阵，它被初始化为具有小的随机值。这个权重可能用于某种复数运算
        self.complex_weight = nn.Parameter(torch.randn(1, self.max_seq_length // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)
        #这两行代码使用正态分布初始化了item_gating和fusion_gating的权重。这是深度学习中常见的权重初始化方法，有助于模型的收敛。
        self.item_gating.weight.data.normal_(mean = 0, std = 0.02)
        self.fusion_gating.weight.data.normal_(mean = 0, std = 0.02)
        
    def contextual_convolution(self, item_emb, feature_emb):
        #Sequence-Level Representation Fusion
       
        feature_fft = torch.fft.rfft(feature_emb, dim=1, norm='ortho')
        item_fft = torch.fft.rfft(item_emb, dim=1, norm='ortho')

        complext_weight = torch.view_as_complex(self.complex_weight)
        item_conv = torch.fft.irfft(item_fft * complext_weight, n = feature_emb.shape[1], dim = 1, norm = 'ortho')
        fusion_conv = torch.fft.irfft(feature_fft * item_fft, n = feature_emb.shape[1], dim = 1, norm = 'ortho')

        item_gate_w = self.item_gating(item_conv)
        fusion_gate_w = self.fusion_gating(fusion_conv)

        contextual_emb = 2 * (item_conv * torch.sigmoid(item_gate_w) + fusion_conv * torch.sigmoid(fusion_gate_w))
        return contextual_emb

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = self.contextual_convolution(self.item_embedding(item_seq), item_emb)
        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        # Loss  optimization
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]


class Improving(KGCN):
    def __init__(self, config, dataset):
        # 使用TedRecDataset的方法获取嵌入
        self.embeddings = dataset.get_embeddings_for_improving()
        # TedRec的初始化，假设TedRec已经被正确初始化
        self.ted_rec = TedRec(config, dataset)
        super().__init__(config, dataset)
        # 这里设置了一个名为 temperature 的属性，它的值从传入的 config 字典中获取。
        # 在深度学习中，温度参数通常用于控制输出分布的平滑程度，尤其是在softmax函数中。
        self.temperature = config['temperature']
        # plm_embedding 可能代表预训练语言模型（Pre-trained Language Model）的嵌入层。
        # 这两行代码创建了两个线性层（nn.Linear），它们通常用于神经网络中的线性变换。
        # 这里它们被用作门控机制（gating mechanism），可能用于控制信息流。
        self.item_gating = nn.Linear(self.hidden_size, 1)
        self.fusion_gating = nn.Linear(self.hidden_size, 1)
        # 初始化了一个名为 MoEAdaptorLayer 的层，这可能是一个“专家混合”（Mixture of Experts）适配器层
        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob'],
            self.max_seq_length
        )
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        # 创建了一个复数权重矩阵，它被初始化为具有小的随机值。这个权重可能用于某种复数运算
        self.complex_weight = nn.Parameter(
            torch.randn(1, self.max_seq_length // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)
        # 这两行代码使用正态分布初始化了item_gating和fusion_gating的权重。这是深度学习中常见的权重初始化方法，有助于模型的收敛。
        self.item_gating.weight.data.normal_(mean=0, std=0.02)
        self.fusion_gating.weight.data.normal_(mean=0, std=0.02)

    def construct_directed_graph(self, user_item_sequences):
       
        # Initialize the vertex set and edge dict
        V = set()
        E = {}

        # Iterate through all user-item sequences
        for sequence in user_item_sequences:
            for i in range(len(sequence) - 1):
                item_i = sequence[i]
                item_j = sequence[i + 1]

                # Add items to the vertex set
                V.add(item_i)
                V.add(item_j)

                # Update the edge dict
                if item_i not in E:
                    E[item_i] = {}
                if item_j not in E[item_i]:
                    E[item_i][item_j] = 0
                E[item_i][item_j] += 1

        # Calculate the transition confidence score for each edge
        for item_i in E:
            for item_j in E[item_i]:
                E[item_i][item_j] /= sum(E[item_i].values())
                if E[item_i][item_j] < 0.1:
                    del E[item_i][item_j]

        #V, E = super(Improving, self).construct_directed_graph(user_item_sequences)
        return V, E


    # 全局信息聚合
    def global_info_aggregation(self, M, V, E, num_layers):
        curr_M = M
        for _ in range(num_layers):
            updated_M = []
            for i in range(curr_M.shape[0]):
                item_i = list(V)[i]
                item_embedding = curr_M[i]
                if item_i in E:
                    neighbor_embeddings = []
                    for neighbor, confidence in E[item_i].items():
                        neighbor_idx = list(V).index(neighbor)
                        neighbor_embedding = curr_M[neighbor_idx]
                        neighbor_embeddings.append(neighbor_embedding * confidence)
                    aggregated_embedding = torch.sum(torch.stack(neighbor_embeddings, dim=0), dim=0)
                    updated_M.append(self.fusion_gating(item_embedding + aggregated_embedding))
                else:
                    updated_M.append(item_embedding)
            curr_M = torch.stack(updated_M, dim=0)

        global_embeddings = []
        for i in range(curr_M.shape[0]):
            global_embeddings.append(curr_M[i])

        global_embeddings = super(Improving, self).global_info_aggregation(M, V, E, num_layers)
        return global_embeddings

    #软聚类
    def soft_clustering(self, global_embeddings):
        # 执行软聚类
        similarity_scores = torch.matmul(global_embeddings, self.cluster_centers.T)  # (n, c)
        cluster_probs = F.softmax(similarity_scores, dim=1)  # (n, c)

        # 计算发现的子序列
        subsequences = torch.einsum('ij,ik->ijk', global_embeddings, cluster_probs)  # (c, n, d)

        return subsequences


    def forward(self, inputs):
        # 假设 'inputs' 包含了用户交互序列
        user_item_sequences = inputs['user_item_sequences']

        # 获取物品的嵌入表示
        item_emb = self.item_embedding(user_item_sequences)

        # 使用TedRecDataset的新方法获取用户交互序列的嵌入表示
        feature_emb = self.dataset.get_embeddings_for_improving(user_item_sequences)
        # 使用TedRec的方法获取融合了文本信息的ID表示
        ted_output = self.ted_rec.contextual_convolution(item_emb, feature_emb)

        # Perform KGCN-based global information aggregation
        V, E = self.construct_directed_graph(inputs['user_item_sequences'])
        global_embeddings = self.global_info_aggregation(self.plm_embedding, V, E, 2)

        # Apply the MoE adaptor layer
        adapted_embeddings = self.moe_adaptor(global_embeddings)

        # Soft clustering
        subsequences = self.soft_clustering(adapted_embeddings)

        # Compute the logits and apply temperature scaling
        logits = self.item_gating(adapted_embeddings) / self.temperature

        return logits


    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_item_e = self.forward(user, pos_item)
        user_e, neg_item_e = self.forward(user, neg_item)

        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)

        predict = torch.cat((pos_item_score, neg_item_score))
        target = torch.zeros(len(user) * 2, dtype=torch.float32).to(self.device)
        target[: len(user)] = 1
        rec_loss = self.bce_loss(predict, target)

        l2_loss = self.l2_loss(user_e, pos_item_e, neg_item_e)
        loss = rec_loss + self.reg_weight * l2_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user_index = interaction[self.USER_ID]
        item_index = torch.tensor(range(self.n_items)).to(self.device)

        user = torch.unsqueeze(user_index, dim=1).repeat(1, item_index.shape[0])
        user = torch.flatten(user)
        item = torch.unsqueeze(item_index, dim=0).repeat(user_index.shape[0], 1)
        item = torch.flatten(item)

        user_e, item_e = self.forward(user, item)
        score = torch.mul(user_e, item_e).sum(dim=1)

        return score.view(-1)
