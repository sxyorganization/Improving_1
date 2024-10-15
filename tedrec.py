import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.layers import TransformerEncoder, VanillaAttention
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType

    
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


class STFTModel(nn.Module):
    def __init__(self, config):
        super(STFTModel, self).__init__()
        self.n_fft = config['n_fft']
        self.hop_length = config['hop_length']
        self.window = torch.hann_window(self.n_fft)
        self.compress_size = config["compress_size"]
        self.compress = nn.Linear(config["hidden_size"], config["compress_size"])
        self.enlarge = nn.Linear(config["compress_size"], config["hidden_size"])

    def forward(self, x):
        x_compress = self.compress(x)
        stft_results = []
        for i in range(self.compress_size):
            stft_result = torch.stft(
                x_compress[:, :, i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window.to(x.device),
                return_complex=True
            )
            stft_results.append(stft_result)
        stft_tensor = torch.stack(stft_results, dim=-1)
        return stft_tensor

    def inverse_transform(self, stft_tensor, original_seq_len):
        istft_results = []
        for i in range(self.compress_size):
            istft_result = torch.istft(
                stft_tensor[..., i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window.to(stft_tensor.device),
                length=original_seq_len
            )
            istft_results.append(istft_result)
        istft_result_en = torch.stack(istft_results, dim=-1)
        reconstructed_tensor = self.enlarge(istft_result_en)
        return reconstructed_tensor


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class TedRec(SASRec):
    """Text-ID fusion approach for sequential recommendation
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.temperature = config['temperature']
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        self.item_gating = nn.Linear(self.hidden_size, 1)
        self.fusion_gating = nn.Linear(self.hidden_size, 1)
        self.item_gating.weight.data.normal_(mean=0, std=0.02)
        self.fusion_gating.weight.data.normal_(mean=0, std=0.02)
        self.complex_weight = nn.Parameter(torch.randn(1, self.max_seq_length // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob'],
            self.max_seq_length
        )

        self.stft_model = STFTModel(config)
        self.n_fft = config['n_fft']
        self.hop_length = config['hop_length']
        self.L = self.max_seq_length
        self.compress_size = config["compress_size"]
        self.item_stft_gating = nn.Linear(self.hidden_size, 1)
        self.fusion_stft_gating = nn.Linear(self.hidden_size, 1)
        self.item_stft_gating.weight.data.normal_(mean=0, std=0.02)
        self.fusion_stft_gating.weight.data.normal_(mean=0, std=0.02)
        self.complex_weight_stft = nn.Parameter(torch.randn(1, (self.n_fft // 2) + 1, 1 + self.L // self.hop_length, self.compress_size, 2,dtype=torch.float32) * 0.02)

        self.res_mh = MultiHeadAttention(config['n_head'], self.hidden_size, config['d_k'], config['d_v'])
        self.stft_weight = config['stft_weight']
        self.res_weight = config['res_weight']
        
    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion
        """
        # ori

        feature_fft = torch.fft.rfft(feature_emb, dim=1, norm='ortho')
        item_fft = torch.fft.rfft(item_emb, dim=1, norm='ortho')
        complext_weight = torch.view_as_complex(self.complex_weight)
        item_conv = torch.fft.irfft(item_fft * complext_weight, n=feature_emb.shape[1], dim=1, norm='ortho')
        fusion_conv = torch.fft.irfft(feature_fft * item_fft, n=feature_emb.shape[1], dim=1, norm='ortho')
        item_gate_w = self.item_gating(item_conv)
        fusion_gate_w = self.fusion_gating(fusion_conv)
        contextual_emb = 2 * (item_conv * torch.sigmoid(item_gate_w) + fusion_conv * torch.sigmoid(fusion_gate_w))

        # new
        item_seq_len = item_emb.shape[1]
        feature_seq_len = feature_emb.shape[1]
        item_stft = self.stft_model(item_emb)
        feature_stft = self.stft_model(feature_emb)
        complext_weight_stft = torch.view_as_complex(self.complex_weight_stft)
        item_conv_stft = self.stft_model.inverse_transform(item_stft * complext_weight_stft, item_seq_len)
        fusion_conv_stft = self.stft_model.inverse_transform(feature_stft * item_stft, feature_seq_len)
        item_stft_gate_w = self.item_stft_gating(item_conv_stft)
        fusion_stft_gate_w = self.fusion_stft_gating(fusion_conv_stft)
        stft_contextual_emb = 2 * (item_conv_stft * torch.sigmoid(item_stft_gate_w) + fusion_conv_stft * torch.sigmoid(fusion_stft_gate_w))

        # res
        res = item_emb + feature_emb + item_emb * feature_emb # 参考deepfm, fm， ffm etc
        res_mh = self.res_mh(res, res, res)

        merge_emb = contextual_emb + self.stft_weight * stft_contextual_emb + self.res_weight * res_mh
        return merge_emb

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
        return scores
