import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool
from torch_geometric.utils import dropout_adj, softmax
from collections import OrderedDict
from einops.layers.torch import Rearrange, Reduce


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()
        self.project_d = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_p = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, d, p):
        d = self.project_d(d)
        p = self.project_p(p)

        a = torch.cat((d, p), 1)
        a = torch.softmax(a, dim=1)
        return a


##########################处理蛋白质的MCNN部分TargetRepresentation##############################

class Conv1dReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


# GCN-CNN based model
class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 128, 3)
            )

        self.linear = nn.Linear(block_num * 128, 128)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x


##############################################################


#######################处理药物SMILES串的扩张卷积部分######################
class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=256, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            nn.ReLU(),
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),

        )
        self.name = name

    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)


class ResDilaCNNBlocks(nn.Module):
    # def __init__(self, feaSize, filterSize, blockNum=5, dropout=0.35, name='ResDilaCNNBlocks'):
    def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.5,
                 name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()  #
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize, filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}",
                                        ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize,
                                                        dropout=dropout))
            # self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(filterSize,dropout=dropout))
        self.name = name
        self.act = nn.ReLU()

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x)  # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1, 2))  # => batchSize × seqLen × filterSize
        x = self.act(x)  # => batchSize × seqLen × filterSize

        # x = self.pool(x.transpose(1, 2))
        x = Reduce('b c t -> b c', 'max')(x)
        return x

class DiffusionModule(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256, num_steps=10, dropout=0.1):
        super(DiffusionModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_steps = num_steps
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 噪声预测网络
        self.noise_predictor = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 特征细化网络
        self.feature_refiner = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 可学习的噪声调度参数
        self.log_alpha = nn.Parameter(torch.zeros(num_steps))
        
    def get_alpha_schedule(self):
        # 使用可学习的参数生成噪声调度
        alpha = torch.sigmoid(self.log_alpha)
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        return alpha, alpha_cumprod
    
    def forward_diffusion(self, x, t, noise=None):
        """前向扩散过程：添加噪声"""
        batch_size = x.shape[0]
        if noise is None:
            noise = torch.randn_like(x)
        
        _, alpha_cumprod = self.get_alpha_schedule()
        alpha_t = alpha_cumprod[t].view(batch_size, 1, 1)
        
        # 添加噪声
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return noisy_x, noise
    
    def reverse_diffusion(self, x, t):
        """反向扩散过程：去噪"""
        batch_size = x.shape[0]
        
        # 时间步编码
        t_embed = self.time_embed(t.float().view(batch_size, 1))
        t_embed = t_embed.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        # 将时间信息与特征拼接
        x_with_time = torch.cat([x, t_embed], dim=-1)
        
        # 预测噪声
        predicted_noise = self.noise_predictor(x_with_time)
        
        # 计算去噪后的特征
        alpha, alpha_cumprod = self.get_alpha_schedule()
        alpha_t = alpha[t].view(batch_size, 1, 1)
        alpha_cumprod_t = alpha_cumprod[t].view(batch_size, 1, 1)
        
        # 去噪公式
        x_denoised = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        return x_denoised, predicted_noise
    
    def forward(self, x, training=True):
        """
        x: [batch_size, seq_len, embed_dim] - 来自embedding_smiles的输出
        """
        batch_size = x.shape[0]
        
        if training:
            # 训练时：执行前向扩散和反向去噪
            # 随机选择时间步
            t = torch.randint(0, self.num_steps, (batch_size,), device=x.device)
            
            # 前向扩散
            noisy_x, true_noise = self.forward_diffusion(x, t)
            
            # 反向去噪
            denoised_x, predicted_noise = self.reverse_diffusion(noisy_x, t)
            
            # 计算扩散损失（用于辅助训练）
            diffusion_loss = F.mse_loss(predicted_noise, true_noise)
            
            # 特征细化：结合原始特征和去噪特征
            refined_features = self.feature_refiner(torch.cat([x, denoised_x], dim=-1))
            
            return refined_features, diffusion_loss
        
        else:
            # 推理时：只进行轻度去噪增强
            # 使用较小的噪声
            t = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            noise = torch.randn_like(x) * 0.1  # 小噪声
            
            noisy_x = x + noise
            denoised_x, _ = self.reverse_diffusion(noisy_x, t)
            
            # 特征细化
            refined_features = self.feature_refiner(torch.cat([x, denoised_x], dim=-1))
            
            return refined_features, None
            
            
            
class ConformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, conv_kernel_size=31, conv_expansion_factor=2, 
                 ff_expansion_factor=4, dropout=0.1):
        super(ConformerBlock, self).__init__()
        
        # Feed Forward Module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion_factor, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-Head Self Attention Module
        self.norm_attn = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        
        # Convolution Module
        # self.conv_module = nn.Sequential(
        #     # nn.LayerNorm(d_model),
        #     # Pointwise Conv
        #     nn.Linear(d_model, 2 * d_model),
        #     nn.GLU(dim=-1),
        #     # Depthwise Conv
        #     nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
        #              padding=(conv_kernel_size - 1) // 2, groups=d_model),
        #     nn.BatchNorm1d(d_model),
        #     nn.SiLU(),
        #     # Pointwise Conv
        #     nn.Conv1d(d_model, d_model, kernel_size=1),
        #     nn.Dropout(dropout)
        # )

        self.conv_module = nn.Sequential(
            # pointwise conv 1×1 代替 Linear
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),  # 在通道维做门控（输入是 [B, C, T]）
            # depthwise conv
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
                      padding=(conv_kernel_size - 1) // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            # pointwise conv
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        # Feed Forward Module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion_factor, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Norm at the end
        self.norm_out = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model] or [batch_size, d_model] for pooled features
        """
        # 如果输入是池化后的特征 [batch_size, d_model]，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
            
        # First Feed Forward
        residual = x
        x = residual + 0.5 * self.ff1(x)
        
        # Multi-Head Self Attention
        residual = x
        x_norm = self.norm_attn(x)
        x_attn, _ = self.mhsa(x_norm, x_norm, x_norm)
        x = residual + self.dropout_attn(x_attn)
        
        # Convolution Module
        residual = x
        x_conv = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x_conv = self.conv_module(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [batch_size, seq_len, d_model]
        x = residual + x_conv
        
        # Second Feed Forward
        residual = x
        x = residual + 0.5 * self.ff2(x)
        
        # Final Layer Norm
        x = self.norm_out(x)
        
        # 如果原始输入是2D的，返回2D
        if x.size(1) == 1:
            x = x.squeeze(1)
            
        return x


class ProteinConformer(nn.Module):
    def __init__(self, d_model=128, n_blocks=2, n_heads=8, conv_kernel_size=31, 
                 ff_expansion_factor=4, dropout=0.1):
        super(ProteinConformer, self).__init__()
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                dropout=dropout
            ) for _ in range(n_blocks)
        ])
        
        # 可选的位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, add_pos_encoding=True):
        """
        x: [batch_size, seq_len, d_model] 来自TargetRepresentation的输出
           或 [batch_size, d_model] 如果已经池化
        """
        # 添加位置编码（如果需要且输入是序列）
        if add_pos_encoding and x.dim() == 3:
            x = self.pos_encoding(x)
        
        # 通过Conformer块
        for conformer in self.conformer_blocks:
            x = conformer(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


#################################################################

class DTA_GCN(torch.nn.Module):
    def __init__(self, block_num=3, vocab_protein_size=26, embedding_size=128, n_output=1, n_filters=32,
                 num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64, embed_dim=128,
                 dropout=0.2):
        super(DTA_GCN, self).__init__()

        print('DTA_GCN Loading ...')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.attention = Attention(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.Linear1 = nn.Linear(128, 128)
        self.Linear2 = nn.Linear(128, 128)
        self.pLinear1 = nn.Linear(128, 128)
        self.pLinear2 = nn.Linear(128, 128)
        self.meanPool2d = nn.AvgPool2d((1, 2), stride=1)


        self.protein_encoder = TargetRepresentation(block_num, vocab_protein_size, embedding_size)
        #####新加的protein_conformer
        self.protein_conformer = ProteinConformer(
            d_model=128,  # 与TargetRepresentation输出维度一致
            n_blocks=2,
            n_heads=8,
            conv_kernel_size=31,
            ff_expansion_factor=4,
            dropout=dropout
        )
        self.embedding_smiles = nn.Embedding(num_features_mol + 1, embed_dim)
        # self.embedding_prot = nn.Embedding(num_features_pro + 1, embed_dim)
        #####加入的扩散模型
        self.diffusion_module = DiffusionModule(embed_dim=embed_dim, hidden_dim=256, num_steps=10, dropout=dropout)
        self.onehot_smi_net = ResDilaCNNBlocks(embed_dim, embed_dim, name='res_compound')
        # self.onehot_prot_net = ResDilaCNNBlocks(embed_dim, embed_dim, name='res_prot')
        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)



    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch, mol_vec = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch, data_mol.smile_vec
        # get protein input
        target_x, target_edge_index, target_weight, target_batch, target_vec = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch, data_pro.protein_vec
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        #x = self.relu(x)

        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index, target_weight)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        embedded_smiles = self.embedding_smiles(mol_vec)
        
        ####加入的扩散模型
        if self.training:
            enhanced_smiles, diffusion_loss = self.diffusion_module(embedded_smiles, training=True)
        else:
            enhanced_smiles, _ = self.diffusion_module(embedded_smiles, training=False)
        
        mol = self.onehot_smi_net(enhanced_smiles)###next is begining,change it
        #mol = self.onehot_smi_net(embedded_smiles)

        # embedding_prot = self.embedding_prot(target_vec)
        # pro = self.onehot_prot_net(embedding_prot)
        # conv_d = self.conv_smiles_1(embedded_smiles)
        # # flatten
        # mol = conv_d.view(-1, 32 * 121)
        # mol = self.fc1_smiles(mol)

        pro = self.protein_encoder(target_vec)
        pro = self.protein_conformer(pro, add_pos_encoding=False)
        
        
        
        d1Learn = self.Linear1(x)
        d1Learn = torch.relu(d1Learn)
        d2Learn = self.Linear2(mol)
        d2Learn = torch.relu(d2Learn)
        x = x.reshape(x.shape[0], x.shape[1], 1)
        mol = mol.reshape(mol.shape[0], mol.shape[1], 1)
        Mprotein = torch.cat((x, mol), 2)
        outputDrug = self.meanPool2d(Mprotein)
        outputDrug = torch.sigmoid_(outputDrug)
        oneOutputDrug = 1 - outputDrug
        outputDrug = outputDrug.squeeze()
        oneOutputDrug = oneOutputDrug.squeeze()
        FWa = d1Learn * outputDrug
        F1_Wa = d2Learn * oneOutputDrug
        x = x.squeeze()
        mol = mol.squeeze()
        Dout = FWa + x + F1_Wa + mol

        p1Learn = self.pLinear1(xt)  # 1
        p1Learn = torch.relu(p1Learn)  # 1
        p2Learn = self.pLinear2(pro)
        p2Learn = torch.relu(p2Learn)

        xt = xt.reshape(xt.shape[0], xt.shape[1], 1)  # 1
        pro = pro.reshape(pro.shape[0], pro.shape[1], 1)
        Mprotein = torch.cat((xt, pro), 2)  # 1
        outputP = self.meanPool2d(Mprotein)
        outputP = torch.sigmoid_(outputP)
        oneOutputP = 1 - outputP
        outputP = outputP.squeeze()
        oneOutputP = oneOutputP.squeeze()
        FWap = p1Learn * outputP  # 1
        F1_Wap = p2Learn * oneOutputP
        xt = xt.squeeze()  # 1
        pro = pro.squeeze()
        Pout = FWap + xt + F1_Wap + pro  # 1


        # print(x.size(), xt.size())
        # concat
        a = self.attention(Dout,Pout)
        emb = torch.stack([Dout,Pout], dim=1)
        a = a.unsqueeze(dim=2)
        emb = (a * emb).reshape(-1, 2 * 128)

        # xc = torch.cat((Dout,Pout), 1)
        # add some dense layers
        xc = self.fc1(emb)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        
        ####扩散模型最后加的代码
        if self.training and 'diffusion_loss' in locals():
            return out, diffusion_loss
        else:
             return out
    
