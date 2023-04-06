import math
import torch
import torch.nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class HGNLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, num_entities, num_relations, bias=True):
        super(HGNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_entities = num_entities

        self.weight_ent = Parameter(torch.empty(in_features, out_features, device=device))
        self.weight_rel = Parameter(torch.empty(in_features, out_features, device=device))

        if bias:
            self.bias = Parameter(torch.empty(out_features))

        self.project = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(out_features, 1, bias=False)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_ent)
        torch.nn.init.xavier_uniform_(self.weight_rel)

        stdv = 1. / math.sqrt(self.weight_ent.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, ent_mat, rel_mat, adjacencies):
        # supports = [torch.spmm(adjacency, ent_mat) for adjacency in adjacencies]
        # supports = torch.cat(supports).view(self.num_relations, self.num_entities, self.in_features)  # R*E*in
        # ent_output = torch.matmul(supports, self.weight_ent)  # (R*E*in, R*in*out)=R*E*out
        #
        # weight = self.project(rel_mat)
        # alpha = torch.softmax(weight, 0).unsqueeze(1)  # R*1*1
        #
        # ent_output = torch.mul(ent_output, alpha)  # (R*E*out, R*1*1)= R*E*out
        # # ent_output = torch.max(ent_output, 0)[0]  # max: [0] return value; [1] return index
        # # ent_output = torch.mean(ent_output, 0)  # mean
        # ent_output = torch.sum(ent_output, 0)  # sum
        # if self.bias is not None:
        #     ent_output = ent_output + self.bias
        # rel_output = torch.mm(rel_mat, self.weight_rel)  # (R*in, in*out) = R*out
        # return ent_output, rel_output

        supports = [torch.spmm(adjacency, ent_mat) for adjacency in adjacencies]
        supports = torch.cat(supports).view(self.num_relations, self.num_entities, self.in_features)  # R*E*in

        weight = self.project(rel_mat)
        # alpha = torch.softmax(weight, 0).unsqueeze(1)  # R*1*1
        alpha = torch.sigmoid(weight).unsqueeze(1)  # R*1*1
        # print(alpha)
        supports = torch.mul(supports, alpha)  # (R*E*in, R*1*1)= R*E*in
        ent_output = torch.matmul(supports, self.weight_ent)  # (R*E*in, R*in*out)=R*E*out
        ent_output = torch.sum(ent_output, 0)  # sum/GCN
        # ent_output = torch.max(ent_output, 0)[0]  # max/pooling: [0] return value; [1] return index
        # ent_output = torch.mean(ent_output, 0)  # mean

        # if self.bias is not None:
        #     ent_output = ent_output + self.bias
        # rel_output = torch.mm(rel_mat, self.weight_rel)  # (R*in, in*out) = R*out
        return ent_output, rel_mat


class HRAN(torch.nn.Module):

    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(HRAN, self).__init__()

        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.reshape_H = 2
        self.reshape_W = ent_dim

        # for CNN
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_height = kwargs["filt_height"]
        self.filt_width = kwargs["filt_width"]
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout(kwargs["feature_map_dropout"])

        self.emb_e = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.emb_r = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        # filt_dim = self.in_channels * self.out_channels * self.filt_height * self.filt_width
        # self.filter = torch.nn.Embedding(data.relations_num, filt_dim, padding_idx=0)
        # self.rel_filt = Parameter(torch.FloatTensor(ent_dim, filt_dim))
        filter_dim = self.in_channels * self.out_channels * self.filt_height * self.filt_width
        self.filter = torch.nn.Embedding(data.relations_num, filter_dim, padding_idx=0)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)

        # for GNN
        self.gc1 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num)
        self.gc2 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num)
        self.drop_rate = 0.2
        self.bn3 = torch.nn.BatchNorm1d(ent_dim)
        self.bn4 = torch.nn.BatchNorm1d(ent_dim)

        # for prediction
        self.bn2 = torch.nn.BatchNorm1d(ent_dim)
        fc_length = (self.reshape_H - self.filt_height + 1) * \
                    (self.reshape_W - self.filt_width + 1) * \
                    self.out_channels
        self.fc = torch.nn.Linear(fc_length, ent_dim)
        self.register_parameter('bias', Parameter(torch.zeros(data.entities_num)))
        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_e.weight.data)
        torch.nn.init.xavier_normal_(self.emb_r.weight.data)
        # torch.nn.init.xavier_normal_(self.rel_filt)
        torch.nn.init.xavier_normal_(self.filter.weight.data)

    def hgn(self, adjacencies):
        # layer 1
        embedded_ent, embedded_rel = self.gc1(self.emb_e.weight, self.emb_r.weight, adjacencies)
        embedded_ent = torch.relu(self.bn3(embedded_ent))
        embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)

        # layer 2
        # embedded_ent, embedded_rel = self.gc2(embedded_ent, embedded_rel, adjacencies)
        # embedded_ent = torch.relu(self.bn4(embedded_ent))
        # embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
        return embedded_ent, embedded_rel

    def forward(self, e1, rel, embedded_ent, embedded_rel):
        # e1_embedded = embedded_ent[e1].reshape(-1, 1, self.reshape_H, self.reshape_W)
        # x = self.bn0(e1_embedded)
        # x = self.inp_drop(x)
        # x = x.permute(1, 0, 2, 3)

        ent_emb = embedded_ent[e1].reshape(-1, 1, self.ent_dim)
        rel_emb = embedded_rel[rel].reshape(-1, 1, self.rel_dim)
        x = torch.cat([ent_emb, rel_emb], 1).reshape(-1, 1, self.reshape_H, self.reshape_W)
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = x.permute(1, 0, 2, 3)

        f = self.filter(rel)
        f = f.reshape(ent_emb.size(0) * self.in_channels * self.out_channels, 1, self.filt_height,
                      self.filt_width)
        x = F.conv2d(x, f, groups=ent_emb.size(0))

        x = x.reshape(ent_emb.size(0), self.out_channels, self.reshape_H - self.filt_height + 1,
                      self.reshape_W - self.filt_width + 1)

        # f = torch.mm(embedded_rel[rel], self.rel_filt)
        # f = f.reshape(e1_embedded.size(0) * self.in_channels * self.out_channels, 1,
        #               self.filt_height, self.filt_width)
        # # f = self.filter(rel).reshape(e1_embedded.size(0) * self.in_channels * self.out_channels, 1,
        # #                              self.filt_height, self.filt_width)
        # x = F.conv2d(x, f, groups=e1_embedded.size(0))
        # x = x.reshape(e1_embedded.size(0), self.out_channels, self.reshape_H - self.filt_height + 1,
        #               self.reshape_W - self.filt_width + 1)

        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_map_drop(x)

        x = x.reshape(ent_emb.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        # x = torch.relu(x)
        x = self.hidden_drop(x)

        x = torch.mm(x, embedded_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
