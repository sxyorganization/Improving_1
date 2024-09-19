# Improving_1
对tedrec.py里添加了Improving模块，里面包括创建用户交互的有向图，全局信息聚合和软聚类;

TedRec.yaml里面配置了参数：
data_args:
  order: TO
但是仍然存在ValueError: The ordering args for sequential recommendation has to be 'TO'

<h1 style="font-size:50px;">7.9修改内容</h1>
在ted部分将使用了ted的方法，将得到的融合了文本信息的id作为improving的嵌入层，但是这部分的代码融合还是有问题，还在修改，
将原本注释掉的ted部分取消注释，接着通过improving的方法实现，在forward里面引入了ted实现。
报错问题分析：<br>
TO是按时间顺序排序的的序列，遇到的 ValueError，这个错误是由于在顺序推荐系统中，数据需要按照时间顺序（Time Order, TO）进行排序。如果在 props/TedRec.yaml 或 props/overall.yaml 配置文件中设置了排序参数，确保它们正确地指定了 'TO'。如果排序参数已经正确设置，那么问题可能出现在 TedRecDataset 类的实现中。
现在的主要问题是对dataset.py进行修改，通过添加data设置为按时间顺序的方法pass掉，SequentialDataset类里面没有这个，可以试一下自己添加一下修改

<h1 style="font-size:50px;">7.25更新代码</h1>
将main的28行修改，按照老师说的修改之后会出现报错，所以对dataset进行了修改，我的思路是把ted方式融合后的输出与improving的嵌入结合起来，使得improving的用户交互的历史项目里的项目id包含文本信息，然后进行全局信息聚合、软聚类....<br>
实际上对所有的代码都进行了一些调整，但是负样本部分还没删除，今天运行代码过程速度太慢没跑出来，不知道具体删除的是哪个部分。
在dataset里面加入了一个模块：<br>
# 新增方法：获取Improving模型所需的嵌入
def get_improving_embeddings(self, user_interaction_sequence):<br>
    # 假设 user_interaction_sequence 是用户交互的项目ID列表<br>
    sequence_embeddings = []<br>
    for item_id in user_interaction_sequence:<br>
        item_embedding = self.plm_embedding(torch.tensor([item_id], dtype=torch.long))<br>
        sequence_embeddings.append(item_embedding)<br>
    sequence_embeddings = torch.cat(sequence_embeddings, dim=0)<br>
    return sequence_embeddings  <br>
<h1 style="font-size:50px;">7.30修改了tedrec.py，上传了修改的代码，improving的我先替换掉了</h1>
    首先我创建了class SSTModel，具体方法还没有写，还在查资料中，其次，我将contextual_convolution的方法改成了对sst之后的两个特征向量进行融合。<br>
    https://blog.csdn.net/weixin_39402231/article/details/126369658是csdn上的基于python的同步压缩变换的描述和一些基础知识。
<h1 style="font-size:50px;">8.1修改了tedrec.py和TedRec.yaml，上传了修改的代码</h1>
1.tedrec里面添加了sst的方法，但是现在出现了一个：TypeError: linear(): argument 'input' (position 1) must be Tensor, not NoneType的报错问题，还在修改中，先把具体的方法上传上来<br>
2.在TedRec.yaml里面添加了新的参数：<br>
window_length: 256，作用：这是每个窗口的长度，即每次进行傅里叶变换时所使用的信号片段的长度。<br>
hop_length: 128，作用：这是窗口之间的步长，即每次移动窗口时的距离。<br>
n_fft: 512，作用：这是傅里叶变换的点数，即每个窗口进行傅里叶变换时使用的点数<br>
原作者的其他参数并没有调整，初步想的是先把写的方法运行出来了再进行调参。
<h1 style="font-size:50px;">8.9修改了tedrec.py，上传了修改的代码</h1>
遇到了RuntimeError: The size of tensor a (102400) must match the size of tensor b (50) at non-singleton dimension 1
<h1 style="font-size:50px;">8.27上传了修改的代码，有一些新的问题</h1>
coeffs[49][0] shape: torch.Size([10, 1])
coeffs[49][1] shape: torch.Size([10, 1])
coeffs[49][2] shape: torch.Size([19, 1])
coeffs[49][3] shape: torch.Size([38, 1])
coeffs[49][4] shape: torch.Size([75, 1])
coeffs[49][5] shape: torch.Size([150, 1])
stacked_coeffs[0] shape: torch.Size([50, 10, 1])
stacked_coeffs[1] shape: torch.Size([50, 10, 1])
stacked_coeffs[2] shape: torch.Size([50, 19, 1])
stacked_coeffs[3] shape: torch.Size([50, 38, 1])
stacked_coeffs[4] shape: torch.Size([50, 75, 1])
stacked_coeffs[5] shape: torch.Size([50, 150, 1])
item_sst shape: torch.Size([50, 10])
feature_sst shape: torch.Size([50, 10])
Traceback (most recent call last):
  File "E:\python\TedRec-main\main.py", line 69, in <module>
    run_model(args.d)
  File "E:\python\TedRec-main\main.py", line 41, in run_model
    best_valid_score, best_valid_result = trainer.fit(
  File "C:\Users\28258\anaconda3\envs\pytorch\lib\site-packages\recbole\trainer\trainer.py", line 439, in fit
    train_loss = self._train_epoch(
  File "C:\Users\28258\anaconda3\envs\pytorch\lib\site-packages\recbole\trainer\trainer.py", line 245, in _train_epoch
    losses = loss_func(interaction)
  File "E:\python\TedRec-main\tedrec.py", line 307, in calculate_loss
    seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
  File "E:\python\TedRec-main\tedrec.py", line 278, in forward
    input_emb.append(self.contextual_convolution(self.item_embedding(item_seq[i]), item_emb_i))
  File "E:\python\TedRec-main\tedrec.py", line 260, in contextual_convolution
    item_gate_w = self.item_gating(item_sst)
  File "C:\Users\28258\anaconda3\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\28258\anaconda3\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\28258\anaconda3\envs\pytorch\lib\site-packages\torch\nn\modules\linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (50x10 and 300x1)<br>
反复循环这些：item_sst shape: torch.Size([2048, 300])
feature_sst shape: torch.Size([2048, 300])
input_emb shape: torch.Size([2048, 50, 300])
position_embedding shape: torch.Size([2048, 50, 300])
但是维度对上了，都一致了。</br>
<h1>9.19一些可以用到写作里的内容，对比傅里叶和小波的内容</h1>
小波分析与傅里叶变换的区别
时间和频率表示：
傅里叶变换：提供全局的频率信息，适合处理周期性信号，但对于非平稳信号（如在时间上变化的频率）表现不佳。
小波变换：提供时间和频率的局部表示，能够捕捉信号的瞬时变化，适合分析非平稳信号。
信号分解方式：
傅里叶变换：将信号分解为正弦波的叠加，无法有效处理瞬态信号。
小波变换：将信号分解为小波基函数的叠加，能更好地捕捉瞬态特征和局部变化。
优缺点比较
小波分析的优点
局部特征提取：能够捕捉信号中的局部变化，适合稀疏性较高的嵌入向量。
去噪能力：在去噪和特征提取方面表现良好，可以分离信号中的重要成分。
多分辨率分析：提供不同尺度的分析，有助于识别多层次的特征。
小波分析的缺点
计算复杂度：相对于傅里叶变换，小波变换的计算复杂度较高，尤其是在高维数据上。
选择小波基和层数：需要谨慎选择小波基和分解层数，以获得最佳效果。
傅里叶变换的优点
计算效率：傅里叶变换的计算效率较高，尤其是使用快速傅里叶变换（FFT）算法时。
全局频率信息：能够提供信号的全局频率特性，适合周期性信号的分析。
傅里叶变换的缺点
无法捕捉瞬时变化：对非平稳信号的分析效果较差，难以捕捉瞬时特征。
分辨率问题：不能在不同频率范围内提供不同的时间分辨率。

