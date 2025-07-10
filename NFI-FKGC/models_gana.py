from embedding import *
from hyper_embedding import *
from collections import OrderedDict
import torch
import json
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class LSTM_attn(nn.Module):
    def __init__(self,device, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5,drop_path=0.0):
        super(LSTM_attn, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True, dropout=self.dropout)
        # self.gru = nn.GRU(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True)
        self.out = nn.Linear(self.n_hidden*2*self.layers, self.out_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fc = nn.Linear(self.embed_size * 2, self.out_size)
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden*2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2).to(self.device)
        # batchnorm = nn.BatchNorm1d(1, affine=False).cuda()
        # attn_weight = batchnorm(attn_weight)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weight)
        context = context.view(-1, self.n_hidden*2*self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        x = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers * 2, size[0], self.n_hidden)).to(self.device)
        cell_state = Variable(torch.zeros(self.layers * 2, size[0], self.n_hidden)).to(self.device)
        output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))  # LSTM
        output = output.permute(1, 0, 2)
        output = output + self.drop_path(output)
        attn_output = self.attention_net(output, final_cell_state)  # change log
        attn_output = attn_output + self.drop_path(attn_output)
        outputs = self.out(attn_output) + self.fc(inputs.mean(dim=1))
        outputs = outputs + self.drop_path(outputs)  # delete6.1
        return outputs.view(size[0], 1, 1, self.out_size)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num,norm,theta):
        norm = norm[:,:1,:,:]
        h = h - torch.sum(h * norm, -1, True) * norm
        t = t - torch.sum(t * norm, -1, True) * norm
        score = -theta.unsqueeze(1)*torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


def save_grad(grad):
    global grad_norm
    grad_norm = grad


class MetaR(nn.Module):
    def __init__(self, dataset, parameter, num_symbols, embed = None):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.rel2id = dataset['rel2id']
        self.num_rel = len(self.rel2id)
        self.embedding = Embedding(dataset, parameter)
        self.h_embedding = H_Embedding(dataset, parameter)
        self.few = parameter['few']
        self.dropout = nn.Dropout(0.5)
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx = num_symbols)
        self.num_hidden1 = 500
        self.num_hidden2 = 200
        self.lstm_dim = parameter['lstm_hiddendim']
        self.lstm_layer = parameter['lstm_layers']
        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))

        self.h_emb = nn.Embedding(self.num_rel, self.embed_dim)
        init.xavier_uniform_(self.h_emb.weight)

        self.Linear1= nn.Linear(self.embed_dim, self.embed_dim, bias=False)  #
        self.Linear2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rel_w = nn.Bilinear(self.embed_dim, self.embed_dim, 2 * self.embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        init.xavier_normal_(self.Linear1.weight)
        init.xavier_normal_(self.Linear2.weight)

        self.symbol_emb.weight.requires_grad = False
        self.h_norm = None

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = LSTM_attn(device=self.device,embed_size=50, n_hidden=self.lstm_dim, out_size=50,layers=self.lstm_layer,
                                              dropout=self.dropout_p,drop_path=0.2)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = LSTM_attn(device=self.device,embed_size=100, n_hidden=self.lstm_dim, out_size=100, layers=self.lstm_layer,
                                              dropout=self.dropout_p,drop_path=0.0)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.norm_q_sharing = dict()
        self.entity_attention_module = Attention(hidden_size=2*self.embed_dim, num_heads=1)


    def neighbor_encoder(self, support_meta, iseval):#V3
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        entity_self_left,relations_left,entities_left = (support_left_connections[:, 0, 0].squeeze(-1),
                                                         support_left_connections[:, :, 1].squeeze(-1),
                                                         support_left_connections[:, :, 2].squeeze(-1))
        entity_self_right, relations_right, entities_right = (support_right_connections[:, 0, 0].squeeze(-1),
                                                           support_right_connections[:, :, 1].squeeze(-1),
                                                           support_right_connections[:, :, 2].squeeze(-1))

        entself_embeds_left,rel_embeds_left,ent_embeds_left = (self.dropout(self.symbol_emb(entity_self_left)),
                                                               self.dropout(self.symbol_emb(relations_left)),
                                                               self.dropout(self.symbol_emb(entities_left)),)

        entself_embeds_right,rel_embeds_right, ent_embeds_right = (self.dropout(self.symbol_emb(entity_self_right)),
                                                                   self.dropout(self.symbol_emb(relations_right)),
                                                                   self.dropout(self.symbol_emb(entities_right)),)

        rel_emb = self.rel_w(entself_embeds_left, entself_embeds_right)
        rel_emb_forward, rel_emb_backward = torch.split(rel_emb,entself_embeds_left.size(-1), dim=-1)

        if iseval:
            rel_emb_forward,rel_emb_backward,entself_embeds_left,entself_embeds_right = (rel_emb_forward.unsqueeze(0),rel_emb_backward.unsqueeze(0),
                                                                                       entself_embeds_left.unsqueeze(0),entself_embeds_right.unsqueeze(0))
        entity_level_left = self.entity_attention_module(rel_emb_backward,entself_embeds_left,rel_embeds_left,ent_embeds_left)
        entity_level_left = torch.relu(self.Linear1(entity_level_left) + self.Linear2(entself_embeds_left))
        enhance_left = self.layer_norm(entity_level_left+entself_embeds_left)

        entity_level_right = self.entity_attention_module(rel_emb_forward,entself_embeds_right,rel_embeds_right,ent_embeds_right)
        entity_level_right = torch.relu(self.Linear1(entity_level_right) + self.Linear2(entself_embeds_right))
        enhance_right = self.layer_norm(entity_level_right + entself_embeds_right)
        return enhance_left,enhance_right

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2


    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False,theta=None):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        norm_vector, rel_idx = self.h_embedding(task[0])
        theta = [theta[i] for i in rel_idx[:, :1, :].view(-1).cpu().numpy()]
        theta = torch.from_numpy(np.asarray(theta, dtype=np.float64)).to(self.device)
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative


        support_left,support_right = self.neighbor_encoder(support_meta[0], iseval)
        support_few = torch.cat((support_left, support_right), dim=-1)
        support_few = support_few.view(support_few.shape[0], 2, self.embed_dim)

        for i in range(self.few-1):
            support_left, support_right = self.neighbor_encoder(support_meta[i+1], iseval)
            support_pair = torch.cat((support_left, support_right), dim=-1)  # tanh
            support_pair = support_pair.view(support_pair.shape[0], 2, self.embed_dim)
            support_few = torch.cat((support_few, support_pair), dim=1)
        support_few = support_few.view(support_few.shape[0], self.few, 2, self.embed_dim)
        rel = self.relation_learner(support_few)
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]

        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few,norm_vector,theta)#

                y = torch.Tensor([1]).to(self.device)
                y = y.unsqueeze(1)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
                norm_q = norm_vector - self.beta*grad_meta# hyper-plane update
            else:
                rel_q = rel
                norm_q = norm_vector

            self.rel_q_sharing[curr_rel] = rel_q
            self.h_norm = norm_vector.mean(0)
            self.h_norm = self.h_norm.unsqueeze(0)

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        if iseval:
            norm_q = self.h_norm
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, norm_q,theta)#

        return p_score, n_score

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads,):
        super(Attention, self).__init__()
        self.n = n = 4
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(n*hidden_size, n*num_heads * att_size, bias=False)
        self.linear_k = nn.Linear(n*hidden_size, n*num_heads * att_size, bias=False)
        self.linear_v = nn.Linear(n*hidden_size, n*num_heads * att_size, bias=False)

        self.att_dropout = nn.Dropout(0.2)
        self.Bilinear_att = nn.Linear(n*self.att_size, n*self.att_size, bias=False)


        self.Mlp_w = nn.Linear(hidden_size, n*hidden_size)

        self.gate_w = nn.Linear(n*hidden_size, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))
        self.output_layer = nn.Linear(n*hidden_size, hidden_size//2)
        self.add_linear = nn.Linear(n*hidden_size, hidden_size//2)
        self.ent_self = nn.Linear(hidden_size // 2, hidden_size // 2)
        init.xavier_normal_(self.Mlp_w.weight)


    def forward(self, rel_emb,entself_embeds, rel_embeds, ent_embeds):
        """
        q (target_rel):  (few/b, 1, dim)
        k (nbr_rel):    (few/b, max, dim)
        v (nbr_ent):    (few/b, max, dim)
        mask:   (few/b, max)
        output:
        """
        n = self.n
        q = torch.relu(self.Mlp_w(torch.cat((rel_emb, entself_embeds), dim=-1)))
        k = v = torch.relu(self.Mlp_w(torch.cat((rel_embeds, ent_embeds), dim=-1)))


        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, n*self.att_size)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, n*self.att_size)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, n*self.att_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)
        v = v.transpose(1, 2)


        x = torch.matmul(self.Bilinear_att(q), k)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1,1, n*self.num_heads * self.att_size).squeeze(2)

        gate_tmp = self.gate_w(x) + self.gate_b
        gate = torch.sigmoid(gate_tmp)
        out_neigh = torch.mul(x, gate)
        out_neigh = self.add_linear(out_neigh)
        x = out_neigh + torch.mul(self.ent_self(entself_embeds.unsqueeze(1)), 1.0 - gate)

        return x.squeeze(1)
