import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyexpat import features
from pytorchcrf import CRF

class ProteinModel(nn.Module):
    def __init__(self, d_model=2560, num_classes=5, hidden_dim=128):
        super(ProteinModel, self).__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)
        )

        self.backbone = TransformerEncoderWrapper(
            d_model=hidden_dim,
            num_layers = 4,
            num_heads = 8,
            d_ff = 2048,
            dropout = 0.2
        )

        self.local_conv = MultiScaleConvWrapper(
            d_model=hidden_dim,
            dropout=0.2
        )

        self.classification = ComplexMLP(
            hidden_dim,
            num_classes,
            [128],
            dropout=0.2
        )

        self.crf_turn = ComplexMLP(
            hidden_dim,
            2,
            [128],
            dropout=0.2
        )
        self.crf = CRF(num_tags=2, batch_first=True)

        self._initialize_transition_matrix()

    def _initialize_transition_matrix(self):
        transition_matrix = self.crf.transitions.data.clone()
        transition_matrix[0, 1] = -1e4
        self.crf.transitions.data.copy_(transition_matrix)

    def forward(self, x, tags=None, return_tsne=False, return_conv_maps=False, ablation_type='full'):
        raw_input_mean = torch.mean(x, dim=1)

        x = self.input_projection(x)

        global_features = self.backbone(x)

        if return_conv_maps:
            features, out3, out5, out9 = self.local_conv(global_features, return_scales=True)
        else:
            features = self.local_conv(global_features)

        # features = self.local_conv(global_features)

        class_input = torch.mean(features, dim=1)
        class_output = self.classification(class_input)

        if return_tsne:
            return raw_input_mean, class_input, class_output

        probs = F.softmax(class_output, dim=1)
        pred_class = torch.argmax(class_output, dim=1)
        hard_gate = (pred_class != 0).float().unsqueeze(1).unsqueeze(1)
        if self.training:
            is_targeting_prob = 1.0 - probs[:, 0]
            gate = is_targeting_prob.unsqueeze(1).unsqueeze(1)
            gated_features = features * (gate + 0.1)
        else:
            gated_features = features * hard_gate

        crf_input = self.crf_turn(gated_features)

        if tags is not None:
            crf_output = -self.crf(crf_input, tags, reduction='mean')
        else:
            crf_output = self.crf.decode(crf_input)

        if return_conv_maps:
            return class_output, crf_output, out3, out5, out9

        return class_output, crf_output


# --- 通道注意力 ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  # 输出 0~1 的权重
        )

    def forward(self, x):
        b, c, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y.expand_as(x)


# --- 多尺度卷积 ---
class MultiScaleConvWrapper(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(MultiScaleConvWrapper, self).__init__()

        # 定义三个不同尺度的卷积核 (3, 5, 9)
        # padding 保证输出长度不变
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, padding=4)

        # 融合层：将3个卷积的结果融合回原始维度
        self.fusion = nn.Linear(d_model * 3, d_model)

        # SE-Block
        self.se = SEBlock(d_model, reduction=16)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, return_scales=False):
        x_trans = x.transpose(1, 2)

        out3 = self.conv3(x_trans).transpose(1, 2)
        out5 = self.conv5(x_trans).transpose(1, 2)
        out9 = self.conv9(x_trans).transpose(1, 2)

        concat = torch.cat([out3, out5, out9], dim=-1)

        out = self.fusion(concat)
        out = self.activation(out)

        out_trans = out.transpose(1, 2)
        out_se = self.se(out_trans)
        out = out_se.transpose(1, 2)

        final_out = self.norm(x + self.dropout(out))

        if return_scales:
            return final_out, out3, out5, out9

        return final_out


# --- Transformer ---
class TransformerEncoderWrapper(nn.Module):
    def __init__(self, d_model, num_layers=4, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerEncoderWrapper, self).__init__()
        # 使用可学习位置编码
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout, max_len=205)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)


# --- 可学习位置编码 ---
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# --- 序列分类 ---
class ComplexMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.05, activation='gelu'):
        super(ComplexMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self._get_activation(activation))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(self._get_activation(activation))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.GELU)

    def forward(self, x):
        return self.layers(x)