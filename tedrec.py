import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.layers import TransformerEncoder, VanillaAttention
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType
from scipy.signal import welch, csd
from torch.utils.tensorboard import SummaryWriter
import pywt
import numpy as np


class DTRLayer(nn.Module):
    """Distinguishable Textual Representations Layer
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
        gates = self.noisy_top_k_gating(x, self.training)  # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]  # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class SSTModel(nn.Module):
    def __init__(self, config):
        super(SSTModel, self).__init__()
        self.window_length = config['window_length']
        self.hop_length = config['hop_length']
        self.n_fft = config['n_fft']
        # self.wavelet = config['wavelet']  # 使用 Daubechies 小波
        self.wavelet = pywt.Wavelet('db1')

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        coeffs = []
        level = 5  # 固定小波分解层数
        max_depth = level  # 最大层数直接设置为 level

        # 记录每层级的最大长度，用于对齐
        max_lengths = [0] * (level + 1)

        # 遍历每个样本
        for i in range(x.size(0)):
            # 对每个样本进行小波分解
            c = pywt.wavedec(x[i].detach().numpy(), self.wavelet, level=level)

            # 将小波系数转换为 PyTorch 张量，并记录每个层级的最大长度
            c_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in c]
            coeffs.append(c_tensors)

            for j in range(len(c_tensors)):
                max_lengths[j] = max(max_lengths[j], c_tensors[j].size(0))

        # 填充或裁剪小波系数，使每一层的长度一致
        for i in range(len(coeffs)):
            for j in range(len(coeffs[i])):
                # 确保每个系数数组的长度与该层级的最大长度对齐
                if coeffs[i][j].size(0) < max_lengths[j]:
                    coeffs[i][j] = F.pad(coeffs[i][j], (0, max_lengths[j] - coeffs[i][j].size(0)))
                elif coeffs[i][j].size(0) > max_lengths[j]:
                    coeffs[i][j] = coeffs[i][j][:max_lengths[j]]

            # 填充使所有样本的分解层数一致
            while len(coeffs[i]) < max_depth + 1:
                coeffs[i].append(torch.zeros(max_lengths[len(coeffs[i])], dtype=torch.float32))

        # 确保每个层级的张量形状一致，处理多维情况（例如 [50, 2] 和 [50, 3]）
        for depth in range(max_depth + 1):
            # 找到当前层级中最大的张量宽度（第二维度）
            max_width = max(coeffs[i][depth].size(1) if coeffs[i][depth].dim() > 1 else 1 for i in range(len(coeffs)))
            for i in range(len(coeffs)):
                # 如果张量是一维的，扩展为二维
                if coeffs[i][depth].dim() == 1:
                    coeffs[i][depth] = coeffs[i][depth].unsqueeze(1)

                # 填充宽度，使得每个张量在该层级的第二个维度一致
                if coeffs[i][depth].size(1) < max_width:
                    padding = max_width - coeffs[i][depth].size(1)
                    coeffs[i][depth] = F.pad(coeffs[i][depth], (0, padding))

        # 检查 coeffs 列表的长度
        max_length = max(len(coeffs[i]) for i in range(len(coeffs)))
        for i in range(len(coeffs)):
            # 填充或裁剪 coeffs[i]，使其长度与 max_length 一致
            if len(coeffs[i]) < max_length:
                coeffs[i] = coeffs[i] + [torch.zeros_like(coeffs[i][0])] * (max_length - len(coeffs[i]))
            elif len(coeffs[i]) > max_length:
                coeffs[i] = coeffs[i][:max_length]

        # 打印调试信息
        for i in range(len(coeffs)):
            for j in range(len(coeffs[i])):
                print(f"coeffs[{i}][{j}] shape: {coeffs[i][j].shape}")

        #将每个子列表中的张量按照形状分组
        grouped_coeffs = [
            [coeffs[i][j] for i in range(len(coeffs))]#每组包含所有子列表中相同层级的小波系数
            for j in range(max_depth + 1)
        ]
        stacked_coeffs = [torch.stack(group) for group in grouped_coeffs]

        # 打印调试信息
        for i in range(len(stacked_coeffs)):
            print(f"stacked_coeffs[{i}] shape: {stacked_coeffs[i].shape}")

        #coeffs_tensor = torch.stack([torch.cat(coeffs[i], dim=1) for i in range(len(coeffs))])

        # 计算瞬时频率（这里需要根据小波系数的具体结构进行调整）
        instantaneous_frequency = torch.diff(torch.angle(stacked_coeffs[0]), dim=-1)
        instantaneous_frequency = F.pad(instantaneous_frequency, (0, 1))

        # 计算同步压缩变换
        sst = torch.zeros_like(stacked_coeffs[0])
        for t in range(sst.shape[-1]):
            for f in range(sst.shape[-2]):
                k = (f + instantaneous_frequency[:, f, t]).long()
                k = torch.clamp(k, 0, sst.shape[-2] - 1)
                sst[:, k, t] += stacked_coeffs[0][:, f, t]

        return sst

class TedRec(SASRec):
    """Text-ID fusion approach for sequential recommendation
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # 这里设置了一个名为 temperature 的属性，它的值从传入的 config 字典中获取。
        # 在深度学习中，温度参数通常用于控制输出分布的平滑程度，尤其是在softmax函数中。
        self.temperature = config['temperature']
        self.hidden_size = config['hidden_size']
        # plm_embedding 可能代表预训练语言模型（Pre-trained Language Model）的嵌入层。
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
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

        self.sst_model = SSTModel(config)  # 引入 SST 模型
        # 创建了一个复数权重矩阵，它被初始化为具有小的随机值。这个权重可能用于某种复数运算
        # self.complex_weight = nn.Parameter(torch.randn(1, self.max_seq_length // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)
        # 这两行代码使用正态分布初始化了item_gating和fusion_gating的权重。这是深度学习中常见的权重初始化方法，有助于模型的收敛。
        self.item_gating.weight.data.normal_(mean=0, std=0.02)
        self.fusion_gating.weight.data.normal_(mean=0, std=0.02)

    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion"""
        # 确保 item_emb 和 feature_emb 都是张量类型
        if not isinstance(item_emb, torch.Tensor):
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
        if not isinstance(feature_emb, torch.Tensor):
            feature_emb = torch.tensor(feature_emb, dtype=torch.float32)

        # 对 item_emb 和 feature_emb 进行 SST 处理
        item_sst = self.sst_model(item_emb)
        feature_sst = self.sst_model(feature_emb)

        # 将 item_sst 和 feature_sst 转换为浮点类型
        item_sst = item_sst.real
        feature_sst = feature_sst.real
        # 将 item_sst 和 feature_sst 变为连续的，然后重新整形为二维张量
        item_sst = item_sst.contiguous().view(item_sst.size(0), -1)
        feature_sst = feature_sst.contiguous().view(feature_sst.size(0), -1)
        # 确保 item_sst 和 feature_sst 的形状与线性层的输入形状匹配
        input_dim = self.item_gating.in_features
        item_sst = item_sst[:, :input_dim]
        feature_sst = feature_sst[:, :input_dim]

        # 门控机制
        item_gate_w = self.item_gating(item_sst)
        fusion_gate_w = self.fusion_gating(feature_sst)

        # 融合时频特征和其他特征
        contextual_emb = 2 * (item_sst * torch.sigmoid(item_gate_w) + feature_sst * torch.sigmoid(fusion_gate_w))
        return contextual_emb

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # 使用contextual_convolution 对每个时间步进行融合
        input_emb = []
        for i in range(item_seq.size(1)):
            # 确保 item_emb[i] 是张量类型
            if not isinstance(item_emb[i], torch.Tensor):
                item_emb[i] = torch.tensor(item_emb[i], dtype=torch.float32)
            input_emb.append(self.contextual_convolution(self.item_embedding(item_seq[:, i]), item_emb[i]))
        input_emb = torch.stack(input_emb, dim=1)

        # 修改位置编码维度
        position_embedding = position_embedding.unsqueeze(1).expand(-1, input_emb.shape[1], -1)
        # input_emb = self.contextual_convolution(self.item_embedding(item_seq), item_emb)

        # 打印形状以调试
        print(f"input_emb shape: {input_emb.shape}")
        print(f"position_embedding shape: {position_embedding.shape}")

        # 确保 position_embedding 的形状与 input_emb 匹配
        if position_embedding.shape[1] != input_emb.shape[1]:
            position_embedding = position_embedding[:, :input_emb.shape[1], :]
        # 修改相加操作
        input_emb = input_emb + position_embedding[:, :input_emb.shape[1], :]
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        # Loss optimization
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
        return scores
