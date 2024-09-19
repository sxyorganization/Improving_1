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
        self.wavelet = config['wavelet']
        self.level = 1  # 固定小波分解层数

    def forward(self, x):
        # 确保输入在正确设备上
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=x.device)

        coeffs = []
        max_lengths = [0] * (self.level + 1)

        # 遍历每个样本进行小波分解，并记录最大长度
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
                    # 使用最大值填充
                    max_value = coeffs[i][j].max().item()
                    coeffs[i][j] = F.pad(coeffs[i][j], (0, max_lengths[j] - length), value=max_value)
                else:
                    coeffs[i][j] = coeffs[i][j][:max_lengths[j]]

            # 填充以确保每个样本的分解层数一致
            while len(coeffs[i]) < self.level + 1:
                # 使用最大值填充缺失的层
                max_value = coeffs[i][0].max().item()  # 假设使用第一层的最大值
                coeffs[i].append(torch.full((max_lengths[len(coeffs[i])],), fill_value=max_value, device=x.device))

        # 处理多维情况，确保每层的张量形状一致
        for depth in range(self.level + 1):
            max_width = max(coeffs[i][depth].size(1) if coeffs[i][depth].dim() > 1 else 1 for i in range(len(coeffs)))
            for i in range(len(coeffs)):
                if coeffs[i][depth].dim() == 1:
                    coeffs[i][depth] = coeffs[i][depth].unsqueeze(1)

                if coeffs[i][depth].size(1) < max_width:
                    padding = max_width - coeffs[i][depth].size(1)
                    coeffs[i][depth] = F.pad(coeffs[i][depth], (0, padding))

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

class TedRec(SASRec):
    """Text-ID fusion approach for sequential recommendation
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.temperature = config['temperature']
        self.hidden_size = config['hidden_size']
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding).to(self.device)  # 确保在正确设备上
        self.item_gating = nn.Linear(self.hidden_size, 1).to(self.device)
        self.fusion_gating = nn.Linear(self.hidden_size, 1).to(self.device)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob'],
            self.max_seq_length
        ).to(self.device)

        self.sst_model = SSTModel(config).to(self.device)
        self.item_gating.weight.data.normal_(mean=0, std=0.02)
        self.fusion_gating.weight.data.normal_(mean=0, std=0.02)

    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion"""
        item_emb = item_emb.to(self.device)  # 确保在正确设备上
        feature_emb = feature_emb.to(self.device)  # 确保在正确设备上

        item_sst = self.sst_model(item_emb)
        feature_sst = self.sst_model(feature_emb)

        item_sst = item_sst.real
        feature_sst = feature_sst.real

        item_sst = item_sst.contiguous().view(item_sst.size(0), -1)
        feature_sst = feature_sst.contiguous().view(feature_sst.size(0), -1)

        input_dim = self.item_gating.in_features
        item_sst = item_sst[:, :input_dim]
        feature_sst = feature_sst[:, :input_dim]

        item_gate_w = self.item_gating(item_sst)  # 确保 item_sst 在正确的设备上
        fusion_gate_w = self.fusion_gating(feature_sst)  # 确保 feature_sst 在正确的设备上

        contextual_emb = 2 * (item_sst * torch.sigmoid(item_gate_w) + feature_sst * torch.sigmoid(fusion_gate_w))
        return contextual_emb

    def forward(self, item_seq, item_emb, item_seq_len):
        item_seq = item_seq.to(self.device)  # 确保在 GPU 上
        item_emb = item_emb.to(self.device)  # 确保在 GPU 上

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=self.device)  # 使用 self.device
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids).to(self.device)  # 确保在 GPU 上
        input_emb = self.contextual_convolution(self.item_embedding(item_seq).to(self.device), item_emb)  # 确保在 GPU 上

        input_emb = input_emb.unsqueeze(1)
        input_emb = input_emb.expand(-1, position_embedding.size(1), -1)

        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq).to(self.device)  # 确保在 GPU 上
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].to(self.device)  # 确保在 GPU 上
        item_seq_len = interaction[self.ITEM_SEQ_LEN].to(self.device)  # 确保在 GPU 上
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq)).to(self.device)  # 确保在 GPU 上

        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.item_embedding.weight.to(self.device)  # 确保在 GPU 上
        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID].to(self.device)  # 确保在 GPU 上
        loss = self.loss_fct(logits, pos_items)
        print(f"Current Loss: {loss.item()}")
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].to(self.device)  # 确保在 GPU 上
        item_seq_len = interaction[self.ITEM_SEQ_LEN].to(self.device)  # 确保在 GPU 上
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq)).to(self.device)  # 确保在 GPU 上

        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.item_embedding.weight.to(self.device)  # 确保在 GPU 上
        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
