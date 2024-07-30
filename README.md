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
<h1 style="font-size:50px;">##7.30修改了tedrec.py，上传了修改的代码，improving的我先替换掉了</h1>
    首先我创建了class SSTModel，具体方法还没有写，还在查资料中，其次，我将contextual_convolution的方法改成了对sst之后的两个特征向量进行融合。<br>
    https://blog.csdn.net/weixin_39402231/article/details/126369658是csdn上的基于python的同步压缩变换的描述和一些基础知识。
