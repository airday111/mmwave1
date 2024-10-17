import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        # 其实就是论文中的根号d_k
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # sz_b: batch_size 批量大小
        # len_q,len_k,len_v: 序列长度 在这里他们都相等
        # n_head: 多头注意力 默认为8
        # d_k,d_v: k v 的dim(维度) 默认都是64
        # 此时q的shape为(sz_b, n_head, len_q, d_k) (sz_b, 8, len_q, 64)
        # 此时k的shape为(sz_b, n_head, len_k, d_k) (sz_b, 8, len_k, 64)
        # 此时v的shape为(sz_b, n_head, len_k, d_v) (sz_b, 8, len_k, 64)
        # q先除以self.temperature(论文中的根号d_k) k交换最后两个维度(这样才可以进行矩阵相乘) 最后两个张量进行矩阵相乘
        # attn的shape为(sz_b, n_head, len_q, len_k)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # 用-1e9代替0 -1e9是一个很大的负数 经过softmax之后接近与0
            # 其一：去除掉各种padding在训练过程中的影响
            # 其二，将输入进行遮盖，避免decoder看到后面要预测的东西。（只用在decoder中）
            attn = attn.masked_fill(mask == 0, -1e9)

        # 先在attn的最后一个维度做softmax 再dropout 得到注意力分数
        attn = self.dropout(F.softmax(attn, dim=-1))
        # 最后attn与v进行矩阵相乘
        # output的shape为(sz_b, 8, len_q, 64)
        output = torch.matmul(attn, v)
        # 返回 output和注意力分数
        return output, attn

# q k v 先经过不同的线性层 再用ScaledDotProductAttention 最后再经过一个线性层
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        # 这里的n_head, d_model, d_k, d_v分别默认为8, 512, 64, 64
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
        # len_q, len_k, len_v 为输入的序列长度
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # 用作残差连接
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q k v 分别经过一个线性层再改变维度
        # 由(sz_b, len_q, n_head*d_k) => (sz_b, len_q, n_head, d_k) (sz_b, len_q, 8*64) => (sz_b, len_q, 8, 64)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 交换维度做attention
        # 由(sz_b, len_q, n_head, d_k) => (sz_b, n_head, len_q, d_k) (sz_b, len_q, 8, 64) => (sz_b, 8, len_q, 64)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # 为head增加一个维度
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # 做attention
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # (sz_b, 8, len_k, 64) => (sz_b, len_k, 8, 64) => (sz_b, len_k, 512)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # 经过fc和dropout
        q = self.dropout(self.fc(q))
        # 残差连接 论文中的Add & Norm中的Add
        q += residual
        # 论文中的Add & Norm中的Norm
        q = self.layer_norm(q)
        # q的shape为(sz_b, len_q, 512)
        # attn的shape为(sz_b, n_head, len_q, len_k)
        return q, attn


# 其实就是一个MLP而已
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        # d_in默认为512 d_hid默认为2048
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):

        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # q k v都是enc_input
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        # enc_output的shape为(sz_b, len_q, 512)
        # enc_slf_attn的shape为(sz_b, n_head, len_q, len_k)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 这里的第一个MultiHeadAttention是带Masked
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # q k v都是dec_input
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        # q是dec_output k和v是enc_output
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        # dec_output的shape为(sz_b, len_q, 512)
        # dec_slf_attn的shape为(sz_b, n_head, len_q, len_k)
        # dec_enc_attn的shape为(sz_b, n_head, len_q, len_k)
        return dec_output, dec_slf_attn, dec_enc_attn



def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()

    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):



        def get_position_angle_vec(position):
            # 长度为512 (hid_j // 2)就是论文中的i
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        # shape为(200, 512)
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # 偶数位置使用sin编码
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        # 奇数位置使用cos编码
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # shape为(1, n_position, d_hid)
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # n_position默认为200 seq_len不会超过200
        # 这里x加入位置编码
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab,
            d_word_vec=128, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=128, d_inner=2048,
            pad_idx=0,
            dropout=0.1, n_position=200, scale_emb=False):
        # n_src_vocab: 源语言词汇表的大小
        # d_word_vec: 词嵌入的维度
        super().__init__()

        # padding_idx如果指定 则padding_idx处的条目不会影响梯度 因此padding_idx 处的嵌入向量在训练期间不会更新 即它仍然是一个固定的"pad"
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        # self.src_word_emb=
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        # Encoder包含了n_layers个EncoderLayer n_layers默认为6
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq,
                # src_mask,
                return_attns=False):
        # src_seq: 输入的序列
        # src_mask: get_pad_mask()得到的结果
        enc_slf_attn_list = []


        # 词嵌入
        # enc_output = self.src_word_emb(src_seq)
        # 本模型中已经词嵌入过了
        enc_output= src_seq # [50,128]
        src_mask=None
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        # 加上位置编码
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)
        # n_layers个EncoderLayer串联在一起
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_trg_vocab,
            d_word_vec=128, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=128, d_inner=2048,
            pad_idx=0,
            n_position=200, dropout=0.1, scale_emb=False):
        # n_trg_vocab: 翻译后语言词汇表的大小
        # d_word_vec: 词嵌入的维度
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq,  enc_output, trg_mask=None, src_mask=None, return_attns=True):
        # trg_seq：翻译后语言序列
        # trg_mask: get_pad_mask()得到的结果和get_subsequent_mask()得到的结果进行与运算（&）
        # enc_output: Encoder的输出
        # src_mask: get_pad_mask()得到的结果
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        # 词嵌入
        # dec_output = self.trg_word_emb(trg_seq)
        dec_output=trg_seq
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        # 加上位置编码
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)
        # n_layers个DecoderLayer串联在一起
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab,
            n_trg_vocab,
            # src_pad_idx, trg_pad_idx,
            d_word_vec=128, d_model=128, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()



        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            # pad_idx=src_pad_idx,
            dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            # pad_idx=trg_pad_idx,
            dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        # Decoder中Embedding层和FC层权重共享
        # Embedding层参数维度是：(v,d)，FC层参数维度是：(d,v)，可以直接共享嘛，还是要转置？其中v是词表大小，d是embedding维度。

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
        # Encoder和Decoder间的Embedding层权重共享
        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight
        self.model_Conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=0, dilation=1, groups=1,
                                bias=True)
        self.model_Conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=0, dilation=1,
                                groups=1, bias=True)
        self.model_Conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=3, padding=0, dilation=1,
                                groups=1, bias=True)
        self.model_Conv4=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=5,stride=2)
        self.model_Conv5=nn.Conv2d(in_channels=4,out_channels=8,kernel_size=5,stride=2)
        self.li=nn.Linear(728,128)
        self.bi=nn.Linear(20,128)


    def forward(self, src_seq,trg_seq):
        src_seq = src_seq.view( src_seq.size(1), 4, 128, 154)
        # src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        # trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_su，bsequent_mask(trg_seq)
        # trg_seq=src_seq
        src_seq1=self.model_Conv1(src_seq)
        src_seq2 = self.model_Conv2(src_seq1)
        src_seq3 = self.model_Conv3(src_seq2)
        src_seq32=torch.reshape(src_seq3,(src_seq3.size(0),1,64,40));
        src_seq34=self.model_Conv4(src_seq32)
        src_seq36=self.model_Conv5(src_seq34)
        src_seq4=torch.reshape(src_seq36,(1,src_seq36.size(0),src_seq36.size(1)*src_seq36.size(2)*src_seq36.size(3)));
        src_seq5=self.li(src_seq4)
        trg_seq1=self.bi(trg_seq)
        enc_output, *_ = self.encoder(src_seq5)

        dec_output, *_ = self.decoder(trg_seq1,  enc_output)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))