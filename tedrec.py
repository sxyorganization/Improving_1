import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.layers import TransformerEncoder, VanillaAttention
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType
from scipy.signal import welch, csd

    
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
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class SSTModel(nn.Module):
    def __init__(self, config):
        super(SSTModel, self).__init__()
        self.window_length =config['window_length']
        self.hop_length = config['hop_length']
        self.n_fft = config['n_fft']
        self.window = torch.hann_window(self.window_length)  # 使用 Hann 窗口，并确保大小匹配 win_length

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # 如果输入是 3D 的，将其转换为 2D
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))

        # 计算短时傅里叶变换
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_length,
                          window=self.window, return_complex=True)

        # 计算瞬时频率
        phase = torch.angle(stft)
        instantaneous_frequency = torch.diff(phase, dim=-1)

        # 计算同步压缩变换
        sst = torch.zeros_like(stft)
        for t in range(stft.shape[-1]):
            for f in range(stft.shape[-2]):
                # 确保 instantaneous_frequency[f, t] 是一个标量
                if instantaneous_frequency[f, t].numel() == 1:
                    k = int(f + instantaneous_frequency[f, t].item())
                else:
                    k = int(f + instantaneous_frequency[f, t][0].item())  # 使用第一个元素
                if 0 <= k < stft.shape[-2]:
                    sst[k, t] += stft[f, t]

        return sst


class TedRec(SASRec):
    """Text-ID fusion approach for sequential recommendation
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        #这里设置了一个名为 temperature 的属性，它的值从传入的 config 字典中获取。
        # 在深度学习中，温度参数通常用于控制输出分布的平滑程度，尤其是在softmax函数中。
        self.temperature = config['temperature']
        self.hidden_size = config['hidden_size']
        #plm_embedding 可能代表预训练语言模型（Pre-trained Language Model）的嵌入层。
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
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

        self.sst_model = SSTModel(config)  # 引入 SST 模型
        #创建了一个复数权重矩阵，它被初始化为具有小的随机值。这个权重可能用于某种复数运算
        #self.complex_weight = nn.Parameter(torch.randn(1, self.max_seq_length // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)
        #这两行代码使用正态分布初始化了item_gating和fusion_gating的权重。这是深度学习中常见的权重初始化方法，有助于模型的收敛。
        self.item_gating.weight.data.normal_(mean = 0, std = 0.02)
        self.fusion_gating.weight.data.normal_(mean = 0, std = 0.02)

    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion"""
        # 对 item_emb 和 feature_emb 进行 SST 处理
        if not isinstance(item_emb, torch.Tensor):
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
        if not isinstance(feature_emb, torch.Tensor):
            feature_emb = torch.tensor(feature_emb, dtype=torch.float32)
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
        input_emb = self.contextual_convolution(self.item_embedding(item_seq), item_emb)
        # 打印形状以调试
        print(f"input_emb shape: {input_emb.shape}")
        print(f"position_embedding shape: {position_embedding.shape}")

        # 确保 position_embedding 的形状与 input_emb 匹配
        if position_embedding.shape[1] != input_emb.shape[1]:
            position_embedding = position_embedding[:, :input_emb.shape[1], :]
        input_emb = input_emb + position_embedding
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
