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
from scipy.signal import hilbert


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

    def __init__(self, n_exps, layers, dropout=0.0, max_seq_length=50, noise=False):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([DTRLayer(layers[0], layers[1], dropout, max_seq_length) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-3):
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
        self.wavelet = config['wavelet']
        self.level = 3  # 固定小波分解层数

    def forward(self, x):
        # 确保输入在正确设备上
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=x.device)

        coeffs = []
        max_lengths = [0] * (self.level + 1)

        # 遍历每个样本进行小波分解
        for i in range(x.size(0)):
            c = pywt.wavedec(x[i].detach().cpu().numpy(), self.wavelet, level=self.level)
            c_tensors = [torch.tensor(arr, dtype=torch.float32, device=x.device) for arr in c]
            coeffs.append(c_tensors)

            for j in range(len(c_tensors)):
                max_lengths[j] = max(max_lengths[j], c_tensors[j].size(0))

        # 填充或裁剪小波系数
        for i in range(len(coeffs)):
            for j in range(len(coeffs[i])):
                length = coeffs[i][j].size(0)
                if length < max_lengths[j]:
                    max_value = coeffs[i][j].max().item()
                    coeffs[i][j] = F.pad(coeffs[i][j], (0, max_lengths[j] - length), value=max_value)
                else:
                    coeffs[i][j] = coeffs[i][j][:max_lengths[j]]

            while len(coeffs[i]) < self.level + 1:
                max_value = coeffs[i][0].max().item()
                coeffs[i].append(torch.full((max_lengths[len(coeffs[i])],), fill_value=max_value, device=x.device))

        coeffs_tensor = torch.stack([torch.cat(coeffs[i], dim=1) for i in range(len(coeffs))])

        # 计算瞬时频率
        instantaneous_frequency = torch.diff(torch.angle(coeffs_tensor), dim=-1)
        instantaneous_frequency = F.pad(instantaneous_frequency, (0, 1))

        # 计算同步压缩变换
        sst = torch.zeros_like(coeffs_tensor, device=x.device)
        for t in range(sst.shape[-1]):
            for f in range(sst.shape[-2]):
                freq_adjusted = torch.clamp(instantaneous_frequency[:, f, t], min=-1, max=1)
                k = (f + freq_adjusted).long()
                k = torch.clamp(k, 0, sst.shape[-2] - 1)
                sst[:, k, t] += coeffs_tensor[:, f, t]

        return sst

    def inverse_transform(self, sst, debug=False):
        reconstructed = []
        num_coeffs = 4  # 小波系数的数量

        for i in range(sst.size(0)):
            sst_numpy = sst[i].detach().cpu().numpy()

            # 确保提取的系数与小波分解一致
            coeffs = [
                sst_numpy[:, :41],  # 逼近系数1
                sst_numpy[:, :41],  # 逼近系数2
                sst_numpy[:, 41:119],  # 细节系数1，调整为正确的索引
                sst_numpy[:, 119:271]  # 细节系数2，调整为正确的索引
            ]

            # 验证系数形状
            for j in range(num_coeffs):
                coeff = coeffs[j]
                if coeff.shape[1] != (41 if j < 2 else (78 if j == 2 else 152)):
                    continue  # 这里可以选择记录错误或抛出异常

            # 进行逆变换
            try:
                reconstructed_sample = pywt.waverec(coeffs, self.wavelet)
                reconstructed_tensor = torch.tensor(reconstructed_sample, dtype=torch.float32, device=sst.device)

                if reconstructed_tensor.shape[0] < sst.shape[1]:
                    reconstructed_tensor = F.pad(reconstructed_tensor,
                                                 (0, sst.shape[1] - reconstructed_tensor.shape[0]), value=0)

                reconstructed.append(reconstructed_tensor)
            except ValueError as e:
                continue  # 这里可以选择记录错误或抛出异常
        if not reconstructed:
            return None

        return torch.stack(reconstructed)

class TedRec(SASRec):
    """Text-ID fusion approach for sequential recommendation"""
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.temperature = config['temperature']
        self.hidden_size = config['hidden_size']
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding).to(self.device)
        self.item_gating = nn.Linear(self.hidden_size, 1).to(self.device)
        self.fusion_gating = nn.Linear(self.hidden_size, 1).to(self.device)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob'],
            self.max_seq_length
        ).to(self.device)

        self.sst_model = SSTModel(config).to(self.device)
        self.item_gating.weight.data.normal_(mean=0, std=0.05)
        self.fusion_gating.weight.data.normal_(mean=0, std=0.05)

    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion"""
        item_emb = item_emb.to(self.device)
        feature_emb = feature_emb.to(self.device)

        # 打印输入嵌入的形状
        #print(f"Item Embedding shape: {item_emb.shape}")
        #print(f"Feature Embedding shape: {feature_emb.shape}")

        item_sst = self.sst_model(item_emb)
        feature_sst = self.sst_model(feature_emb)

        #print(f"Item SST shape: {item_sst.shape}")
        #print(f"Feature SST shape: {feature_sst.shape}")
        # 检查形状一致性
        if item_sst.shape != feature_sst.shape:
            raise ValueError("Item SST and Feature SST shapes do not match!")

        # 将频域特征转换回时域
        item_time = self.sst_model.inverse_transform(item_sst)
        feature_time = self.sst_model.inverse_transform(feature_sst)

        if item_time is None or feature_time is None:
            print("Warning: Inverse transform returned None.")
            return None  # 或者处理错误情况

        # 确保维度一致，必要时进行裁剪
        #input_dim = self.item_gating.in_features
        #item_time = item_time[:, :input_dim]
        #feature_time = feature_time[:, :input_dim]

        # 打印逆变换的结果
        #print(f"Item Time shape: {None if item_time is None else item_time.shape}")
        #print(f"Feature Time shape: {None if feature_time is None else feature_time.shape}")

        # 只使用前300个特征（根据你的情况）
        #item_sst_reduced = item_sst[:, :, :self.hidden_size]
        #feature_sst_reduced = feature_sst[:, :, :self.hidden_size]
        item_gate_w = self.item_gating(item_emb)
        fusion_gate_w = self.fusion_gating(feature_emb)
        #item_gate_w = self.item_gating(item_sst_reduced.view(-1, self.hidden_size))  # reshape 为 (batch_size * seq_len, hidden_size)
        #fusion_gate_w = self.fusion_gating(feature_sst_reduced.view(-1, self.hidden_size))

        # 重新reshape回原来的形状
        #item_gate_w = item_gate_w.view(item_sst.shape[0], item_sst.shape[1], -1)
        #fusion_gate_w = fusion_gate_w.view(feature_sst.shape[0], feature_sst.shape[1], -1)

        # 结合特征
        # 加权结合特征，使用 ReLU 激活
        # 使用Softmax进行注意力权重的计算
        attention_weights = F.softmax(item_gate_w + fusion_gate_w, dim=-1)
        contextual_emb = attention_weights * item_time + (1 - attention_weights) * feature_time
        #contextual_emb = F.relu(item_time * item_gate_w + feature_time * fusion_gate_w)*2
        #contextual_emb = 2 * (item_time * torch.sigmoid(item_gate_w) +feature_time * torch.sigmoid(fusion_gate_w))

        return contextual_emb

    def forward(self, item_seq, item_emb, item_seq_len):
        item_seq = item_seq.to(self.device)
        item_emb = item_emb.to(self.device)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids).to(self.device)

        input_emb = self.contextual_convolution(self.item_embedding(item_seq), item_emb)

        # 直接将 input_emb 与 position_embedding 相加，确保维度匹配
        input_emb = input_emb + position_embedding  # 这一步可能不需要 unsqueeze 和 expand
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq).to(self.device)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].to(self.device)
        item_seq_len = interaction[self.ITEM_SEQ_LEN].to(self.device)
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq)).to(self.device)

        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.item_embedding.weight.to(self.device)
        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID].to(self.device)

        ce_loss = self.loss_fct(logits, pos_items)
        # 随机负采样
        num_neg_samples = 100  # 可以调整为合适的数量
        neg_items = torch.randint(0, test_item_emb.size(0), (logits.size(0), num_neg_samples), device=self.device)

        # 负样本的相似性计算
        neg_logits = torch.matmul(seq_output, test_item_emb[neg_items].transpose(1, 2)) / self.temperature

        # 对比损失：正样本和负样本
        contrastive_loss = -F.logsigmoid(logits.gather(1, pos_items.view(-1, 1))).mean() - F.logsigmoid(
            -neg_logits).mean()

        # 总损失 = 交叉熵损失 + 对比损失
        loss = ce_loss + 0.1 * contrastive_loss  # 0.5 权重可以调整

        print(f"Current Loss: {loss.item()}")
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].to(self.device)
        item_seq_len = interaction[self.ITEM_SEQ_LEN].to(self.device)
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq)).to(self.device)

        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.item_embedding.weight.to(self.device)
        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
