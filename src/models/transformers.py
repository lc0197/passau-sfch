import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from global_vars import device

SINUSOID = 'sinus'
LEARNABLE = 'learn'
SINUISOD_LEARNABLE = 'learn_sinus'
EMB_TYPES = [SINUSOID, LEARNABLE, SINUISOD_LEARNABLE]


class FFN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, input_seq):
        return self.linear2(self.relu(self.linear1(input_seq)))


class GatedFusion(torch.nn.Module):
    def __init__(self, input_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Wa = nn.Linear(input_dim, input_dim)
        self.Wb = nn.Linear(input_dim, input_dim)
        self.Wcomb = nn.Linear(2*input_dim, input_dim)

    def forward(self, input_a, input_b):
        ha = F.tanh(self.Wa(input_a)) # BS, SL, dim
        hb = F.tanh(self.Wb(input_b)) # BS, SL, dim
        comb = torch.concatenate([input_a, input_b], dim=-1) # BS, SL, 2*dim
        z = F.sigmoid(self.Wcomb(comb)) # BS, SL, dim
        h = z * ha + (1-z) * hb
        return h


class CustomTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.ffn = FFN(input_dim, hidden_dim)
        self.do1 = torch.nn.Dropout(dropout)
        self.do2 = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask, attn_mask):
        x, att = self.mha(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.do1(x) + value
        x = self.norm1(x)
        x_ff = self.ffn(x)
        x = x + self.do2(x_ff)
        x = self.norm2(x)
        return x


def indices_from_mask(mask):
    #print(mask.get_device())
    full_row = torch.range(1, mask.shape[-1]).to(device)
    #print(full_row.get_device())
    full_indices = torch.vstack([full_row]*mask.shape[0])
    #print(full_indices.get_device())
    masked = full_indices * mask
    return masked.long()


def get_3d_mask(mask, num_heads, context_size=None):
    # mask is initially BS, SL
    bs = mask.shape[0]
    sl = mask.shape[1]
    if not context_size is None:
        context_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[1])) # SL, SL
        seq_lens = torch.sum(mask, dim=1).cpu().numpy()
        for i in range(context_mask.shape[0]):
            seq_len = int(seq_lens[i])
            for j in range(seq_len):
                context_mask[i, j, max(0, j - context_size): j + context_size+1] = 1.
            # padded part: allow attention to all elements to avoid nan
            context_mask[i, seq_len:, :] = 1.
        #context_mask = np.vstack([np.expand_dims(context_mask, 0)] * bs) # BS, SL, SL
        context_mask = torch.Tensor(context_mask).to(device)
        mask3d = torch.repeat_interleave(mask, sl, dim=0)
        mask3d = torch.reshape(mask3d, (bs, sl, sl))  # BS, SL, SL
        mask3d = mask3d * context_mask # BS, SL, SL
    #mask3d = mask.unsqueeze(-1) # BS, SL, 1
    else:
        # TODO move up
        mask3d = torch.repeat_interleave(mask, sl, dim=0) # BS, SL*SL
        mask3d = torch.reshape(mask3d, (bs, sl, sl)) # BS, SL, SL
    mask3d = torch.repeat_interleave(mask3d, num_heads, dim=0) # BS*NH, SL, SL
    #control = torch.sum(mask3d, dim=-1) # just for debugging
    #control = torch.mean(control, dim=-1)
    return mask3d


# from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
def create_positional_embeddings_matrix(max_seq_len, dim, n=10000, learnable=False):
    matrix = np.zeros((max_seq_len+1, dim)) # +1 because of zero padding
    for k in range(max_seq_len):
        for i in np.arange(int(dim/2)):
            denominator = np.power(n, 2*i/dim)
            matrix[k, 2*i] = np.sin(k/denominator)
            matrix[k, 2*i+1] = np.cos(k/denominator)
    return torch.nn.Embedding(num_embeddings=matrix.shape[0], embedding_dim = dim, padding_idx=0, _freeze= not learnable)


MULT = 'mult'
V_FOCUS = 'v_focused'

MODEL_TYPES = [MULT, V_FOCUS]

def get_model_class(model_type):
    if model_type == V_FOCUS:
        return V_Focused_CNN
    elif model_type == MULT:
        return MulT
    else:
        raise NotImplementedError()

class MulTCM(nn.Module):

    def __init__(self, dim, num_layers, num_heads, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(input_dim=dim, hidden_dim=2*dim, num_heads=num_heads, dropout=dropout)
        ] * num_layers)

    def forward(self, query, key, value, key_padding_mask, attn_mask):
        prev = self.layers[0](query=query, key=key, value=value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if len(self.layers) > 1:
            for i in range(1, len(self.layers)):
                # query is always the same!
                new_enc = self.layers[i](query=query, key=prev, value=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                prev = new_enc + prev
        return prev

class MulT(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tanh = nn.Tanh()
        self.context_size = params.context_window
        self.num_heads = params.trf_num_heads

        self.a_cnn = nn.Conv1d(params.a_dim, params.trf_model_dim, kernel_size=3, padding='same', bias=False)
        self.t_cnn = nn.Conv1d(params.t_dim, params.trf_model_dim, kernel_size=3, padding='same', bias=False)
        self.v_cnn = nn.Conv1d(params.v_dim, params.trf_model_dim, kernel_size=3, padding='same', bias=False)

        self.pos_v = create_positional_embeddings_matrix(max_seq_len=params.max_length, dim=params.trf_model_dim,
                                                         learnable=False)
        self.pos_a = self.pos_v
        self.pos_t = self.pos_v

        assert params.trf_num_at_layers > 0
        assert params.trf_num_v_layers > 0

        self.a2t = MulTCM(dim=params.trf_model_dim, num_layers=params.trf_num_at_layers, num_heads=params.trf_num_heads)
        self.v2t = MulTCM(dim=params.trf_model_dim, num_layers=params.trf_num_at_layers, num_heads=params.trf_num_heads)
        self.t2a = MulTCM(dim=params.trf_model_dim, num_layers=params.trf_num_at_layers, num_heads=params.trf_num_heads)
        self.v2a = MulTCM(dim=params.trf_model_dim, num_layers=params.trf_num_at_layers, num_heads=params.trf_num_heads)
        self.a2v = MulTCM(dim=params.trf_model_dim, num_layers=params.trf_num_v_layers, num_heads=params.trf_num_heads)
        self.t2v = MulTCM(dim=params.trf_model_dim, num_layers=params.trf_num_v_layers, num_heads=params.trf_num_heads)

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(6 * params.trf_model_dim, 1)

        self.pooling = nn.MaxPool2d(kernel_size=(4, 1), stride=(2, 1))

    def forward(self, v: torch.Tensor, a, t, mask):
        v = torch.transpose(v, 1, 2) # BS, V, SL
        v = self.v_cnn(v) # BS, dim, SL
        v = torch.transpose(v, 1, 2) # BS, SL, dim
        v = self.tanh(v)

        a = torch.transpose(a, 1, 2)  # BS, A, SL
        a = self.a_cnn(a)  # BS, dim, SL
        a = torch.transpose(a, 1, 2)  # BS, SL, dim
        a = self.tanh(a)

        t = torch.transpose(t, 1, 2)  # BS, V, SL
        t = self.t_cnn(t)  # BS, dim, SL
        t = torch.transpose(t, 1, 2)  # BS, SL, dim
        t = self.tanh(t)

        emb_indices = indices_from_mask(mask)
        v_pos = self.pos_v(emb_indices)
        a_pos = self.pos_a(emb_indices)
        t_pos = self.pos_t(emb_indices)
        v = v + v_pos
        a = a + a_pos
        t = t + t_pos

        trf_key_mask = ~mask.bool()
        trf_3d_mask = ~get_3d_mask(mask, self.num_heads, context_size=self.context_size).bool()

        a2t = self.a2t(query = a, key=t, value=t, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        a2v = self.a2v(query=a, key=v, value=v, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        t2a = self.a2t(query=t, key=a, value=a, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        t2v = self.a2v(query=t, key=v, value=v, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        v2a = self.a2t(query=v, key=a, value=a, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        v2t = self.a2v(query=v, key=t, value=t, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)

        representation = torch.concatenate([a2t, a2v, t2a, t2v, v2a, v2t], dim=-1) # BS, SL, 6*dim
        representation = self.dropout(self.pooling(representation))

        return self.classification(representation)


class V_Focused_CNN(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_cnn = nn.Conv1d(params.a_dim, params.trf_model_dim, kernel_size=3, padding='same', bias=False)
        self.t_cnn = nn.Conv1d(params.t_dim, params.trf_model_dim, kernel_size=3, padding='same', bias=False)
        self.v_cnn = nn.Conv1d(params.v_dim, params.trf_model_dim, kernel_size=3, padding='same', bias=False)
        self.tanh = nn.Tanh()

        self.num_heads = params.trf_num_heads

        self.context_size = params.context_window

        if params.trf_pos_emb == LEARNABLE:
            # now with weight sharing
            self.pos_v = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_a = self.pos_v
            self.pos_t = self.pos_v
            #self.pos_a = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            #self.pos_t = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
        elif params.trf_pos_emb in [SINUSOID, SINUISOD_LEARNABLE]:
            self.pos_v = create_positional_embeddings_matrix(max_seq_len=params.max_length, dim=params.trf_model_dim,
                                                             learnable= params.trf_pos_emb == SINUISOD_LEARNABLE)
            self.pos_a = self.pos_v
            self.pos_t = self.pos_v


        if params.trf_num_v_layers > 0:
            v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
            self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.trf_num_v_layers)
        else:
            self.v_transformer = None

        if params.trf_num_at_layers > 0:
            a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
            self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.trf_num_at_layers)
            t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                         batch_first=True, dim_feedforward=2*params.trf_model_dim)
            self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.trf_num_at_layers)
        else:
            self.a_transformer = None
            self.t_transformer = None

        self.at_fusion = GatedFusion(input_dim = params.trf_model_dim)

        self.v2at_transformer = CustomTransformerEncoderLayer(input_dim = params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1, hidden_dim=2*params.trf_model_dim)
        self.at2v_transformer = CustomTransformerEncoderLayer(input_dim=params.trf_model_dim,
                                                              num_heads=params.trf_num_heads, dropout=0.1, hidden_dim=2*params.trf_model_dim)

        # takes in v, v2at, at2v and at
        if params.trf_num_mm_layers > 0:
            mm_transformer_layer = nn.TransformerEncoderLayer(d_model=4*params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=8*params.trf_model_dim)
            self.mm_transformer = nn.TransformerEncoder(mm_transformer_layer, num_layers=params.trf_num_mm_layers)
        else:
            self.mm_transformer = None


        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(4*params.trf_model_dim, 1)

        #self.classification_aux = nn.Linear(3*params.trf_model_dim, 9) if params.aux_weight > 0 else None

        self.pooling = nn.MaxPool2d(kernel_size=(4,1), stride=(2,1))


    def forward(self, v:torch.Tensor, a, t, mask):
        #print(v.get_device())
        #print(a.get_device())
        #print(t.get_device())
        #print(self.a_projection.weight.get_device())
        #print(self.a_projection.bias.get_device())
        #lengths = torch.sum(mask.detach(), dim=1)
        v = torch.transpose(v, 1, 2)  # BS, V, SL
        v = self.v_cnn(v)  # BS, dim, SL
        v = torch.transpose(v, 1, 2)  # BS, SL, dim
        v = self.tanh(v)

        a = torch.transpose(a, 1, 2)  # BS, A, SL
        a = self.a_cnn(a)  # BS, dim, SL
        a = torch.transpose(a, 1, 2)  # BS, SL, dim
        a = self.tanh(a)

        t = torch.transpose(t, 1, 2)  # BS, V, SL
        t = self.t_cnn(t)  # BS, dim, SL
        t = torch.transpose(t, 1, 2)  # BS, SL, dim
        t = self.tanh(t)

        emb_indices = indices_from_mask(mask)
        v_pos = self.pos_v(emb_indices)
        a_pos = self.pos_a(emb_indices)
        t_pos = self.pos_t(emb_indices)
        v = v + v_pos
        a = a + a_pos
        t = t + t_pos


        trf_key_mask = ~mask.bool()
        trf_3d_mask = ~get_3d_mask(mask, self.num_heads, context_size=self.context_size).bool()
        if not self.v_transformer is None:
            v = self.v_transformer(v, src_key_padding_mask = trf_key_mask, mask =trf_3d_mask) # BS, SL, dim
        if not self.a_transformer is None:
            a = self.a_transformer(a, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL ,dim
            t = self.t_transformer(t, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL, dim

        at = self.at_fusion(a, t) # BS, SL, dim
        v2at = self.v2at_transformer(query=v, key=at, value=at, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask) # BS, SL, dim
        at2v = self.at2v_transformer(query=at, key=v, value=v, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)

        representation = torch.concatenate([v,at2v, v2at, at], dim=-1) # BS, SL, 2*dim

        if not self.mm_transformer is None:
            representation = self.mm_transformer(representation, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask)

        representation = self.dropout(self.pooling(representation))

        return self.classification(representation)