from collections import defaultdict

import numpy as np
import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class Data:

    def __init__(self, data_dir, reverse=False):
        self.data_name = data_dir[5:-1]

        self.train_data, self.train_data_num, self.train_hrt = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data, self.valid_data_num, self.valid_hrt = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data, self.test_data_num, self.test_hrt = self.load_data(data_dir, "test", reverse=reverse)
        self.all_data = self.train_data + self.valid_data + self.test_data
        self.all_hrt = self.train_hrt + self.valid_hrt + self.test_hrt

        self.entities, self.entities_id, self.entities_num = self.get_entities(self.all_data)
        self.relations, self.relations_id, self.relations_num = self.get_relations(self.all_data)

        self.train_data_id = self.data_id(self.train_data)
        self.valid_data_id = self.data_id(self.valid_data)
        self.test_data_id = self.data_id(self.test_data)
        self.all_data_id = self.data_id(self.all_data)

        self.train_hr_dict = self.get_hr_dict(self.train_data_id)
        self.train_hr_list = list(self.train_hr_dict.keys())
        self.train_hr_list_num = len(self.train_hr_list)
        self.all_hr_dict = self.get_hr_dict(self.all_data_id)

        print('数据集: {}'.format(self.data_name))
        print('实体数: {}'.format(self.entities_num), '关系数: {}'.format(self.relations_num))
        print('训练集: {}'.format(self.train_data_num),
              '验证集: {}'.format(self.valid_data_num),
              '测试集: {}'.format(self.test_data_num))

        # # relations category for WN18
        # # 1-1:  ('_similar_to': 13), ('_verb_group': 17)
        # self.rela13 = [i for i in self.test_data_id if i[1] == 26] + [i for i in self.test_data_id if i[1] == 27]
        # self.rela17 = [i for i in self.test_data_id if i[1] == 34] + [i for i in self.test_data_id if i[1] == 35]
        #
        # # 1-N:  ('_has_part': 2), ('_hyponym': 4), ('_instance_hyponym': 6), ('_member_meronym': 8),
        # #       ('_member_of_domain_region': 9), ('_member_of_domain_topic': 10), ('_member_of_domain_usage': 11),
        # self.rela2 = [i for i in self.test_data_id if i[1] == 4] + [i for i in self.test_data_id if i[1] == 5]
        # self.rela4 = [i for i in self.test_data_id if i[1] == 8] + [i for i in self.test_data_id if i[1] == 9]
        # self.rela6 = [i for i in self.test_data_id if i[1] == 12] + [i for i in self.test_data_id if i[1] == 13]
        # self.rela8 = [i for i in self.test_data_id if i[1] == 16] + [i for i in self.test_data_id if i[1] == 17]
        # self.rela9 = [i for i in self.test_data_id if i[1] == 18] + [i for i in self.test_data_id if i[1] == 19]
        # self.rela10 = [i for i in self.test_data_id if i[1] == 20] + [i for i in self.test_data_id if i[1] == 21]
        # self.rela11 = [i for i in self.test_data_id if i[1] == 22] + [i for i in self.test_data_id if i[1] == 23]
        #
        # # N-1:  ('_hypernym': 3), ('_instance_hypernym': 5), ('_member_holonym': 7), ('_part_of': 12),
        # #       ('_synset_domain_region_of': 14), ('_synset_domain_topic_of': 15), ('_synset_domain_usage_of': 16)
        # self.rela3 = [i for i in self.test_data_id if i[1] == 6] + [i for i in self.test_data_id if i[1] == 7]
        # self.rela5 = [i for i in self.test_data_id if i[1] == 10] + [i for i in self.test_data_id if i[1] == 11]
        # self.rela7 = [i for i in self.test_data_id if i[1] == 14] + [i for i in self.test_data_id if i[1] == 15]
        # self.rela12 = [i for i in self.test_data_id if i[1] == 24] + [i for i in self.test_data_id if i[1] == 25]
        # self.rela14 = [i for i in self.test_data_id if i[1] == 28] + [i for i in self.test_data_id if i[1] == 29]
        # self.rela15 = [i for i in self.test_data_id if i[1] == 30] + [i for i in self.test_data_id if i[1] == 31]
        # self.rela16 = [i for i in self.test_data_id if i[1] == 32] + [i for i in self.test_data_id if i[1] == 33]
        #
        # # N-N   ('_also_see': 0), ('_derivationally_related_form': 1)
        # self.rela0 = [i for i in self.test_data_id if i[1] == 0] + [i for i in self.test_data_id if i[1] == 1]
        # self.rela1 = [i for i in self.test_data_id if i[1] == 2] + [i for i in self.test_data_id if i[1] == 3]

    @staticmethod
    def normalization(adjacencies):
        sparse_to_dense = adjacencies.to_dense()
        degrees = [[len(i.nonzero())] for i in sparse_to_dense]

        # D^-1
        # degrees_rec = torch.FloatTensor(degrees).reciprocal()
        # nor_adj = torch.mul(sparse_to_dense, degrees_rec).to_sparse()

        # D^-0.5
        degree_rsq = torch.FloatTensor(degrees).rsqrt()
        nor_adj = torch.mul(torch.mul(degree_rsq, sparse_to_dense), degree_rsq).to_sparse()
        return nor_adj

    def get_adjacencies(self, alpha):
        """
        :param alpha: alpha
        :return: list[tensor]: Adjacency matrix R*E*E
        """
        # for SACN
        # rows, columns, values = [], [], []
        # for h, r, t in self.train_data_id:
        #     rows.append(h), columns.append(t), values.append(r)
        # rows = rows + [i for i in range(self.entities_num)]
        # columns = columns + [i for i in range(self.entities_num)]
        # values = values + [self.relations_num for i in range(self.entities_num)]
        # indices = torch.LongTensor([rows, columns]).to(device)
        # values = torch.LongTensor(values).to(device)
        # adjacencies = [indices, values]

        # for HRAN
        adjacencies = []
        dia_rows = dia_columns = [i for i in range(self.entities_num)]
        dia_value = [alpha for i in range(self.entities_num)]

        for i in range(self.relations_num):
            degrees = [[1] for i in range(self.entities_num)]
            rows, columns, values = [], [], []
            for h, r, t in self.train_data_id:
                if i == r:
                    degrees[h][0] += 1
                    rows.append(h)
                    columns.append(t)
                    values.append(1.0 - alpha)
            rows = rows + dia_rows
            columns = columns + dia_columns
            values = values + dia_value
            sparse_matrix = torch.sparse_coo_tensor(torch.LongTensor([rows, columns]),
                                                    torch.FloatTensor(values),
                                                    [self.entities_num, self.entities_num])
            # D^-1
            # degrees_rec = torch.FloatTensor(degrees).reciprocal().repeat(1, self.entities_num).to_sparse() # E*1  E*E
            # nor_adj = torch.mul(sparse_to_dense, degrees_rec).to_sparse()

            # D^-0.5
            # degrees_rsq = torch.FloatTensor(degrees).rsqrt().repeat(1, self.entities_num).to_sparse()  # E*1  E*E
            # sparse_matrix = torch.mul(torch.mul(degrees_rsq, sparse_matrix), degrees_rsq).to(device)
            sparse_matrix = self.normalization(sparse_matrix).to(device)
            adjacencies.append(sparse_matrix)
        return adjacencies

    @staticmethod
    def get_hr_dict(data_id):
        hr_dict = defaultdict(list)
        for triple in data_id:
            hr_dict[(triple[0], triple[1])].append(triple[2])
        return hr_dict

    def get_batch_train_data(self, batch_size):
        start = 0
        np.random.shuffle(self.train_hr_list)
        while start < self.train_hr_list_num:
            end = min(start + batch_size, self.train_hr_list_num)
            batch_data = self.train_hr_list[start:end]

            batch_target = np.zeros((len(batch_data), self.entities_num))
            for index, hr_pair in enumerate(batch_data):
                batch_target[index, self.train_hr_dict[hr_pair]] = 1.0

            batch_data = torch.tensor(batch_data)
            batch_target = torch.FloatTensor(batch_target)

            start = end
            yield batch_data, batch_target

    @staticmethod
    def get_batch_eval_data(batch_size, eval_data):
        eval_data_num = len(eval_data)
        start = 0
        while start < eval_data_num:
            end = min(start + batch_size, eval_data_num)

            batch_data = eval_data[start:end]
            batch_num = len(batch_data)
            batch_data = torch.tensor(batch_data)

            start = end
            yield batch_data, batch_num

    def data_id(self, data):
        data_num = len(data)
        data_id = [(self.entities_id[data[i][0]],
                    self.relations_id[data[i][1]],
                    self.entities_id[data[i][2]])
                   for i in range(data_num)]
        return data_id

    @staticmethod
    def load_data(data_dir, data_type, reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            hrt = f.read().strip().split("\n")
            hrt = [i.split() for i in hrt]
            trh = []
            if reverse:
                trh = [[i[2], i[1] + "_reverse", i[0]] for i in hrt]
            data = hrt + trh
            data_num = len(data)
            f.close()
        return data, data_num, hrt

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        entities_num = len(entities)
        entities_id = {entities[i]: i for i in range(entities_num)}
        return entities, entities_id, entities_num

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        relations_num = len(relations)
        relations_id = {relations[i]: i for i in range(relations_num)}
        return relations, relations_id, relations_num

    def get_complex_triple(self):
        relation_to_head = defaultdict(set)
        relation_to_tail = defaultdict(set)
        relation_to_trip = defaultdict(int)

        for h, r, t in self.all_hrt:
            relation_to_head[r].add(h)
            relation_to_tail[r].add(t)
            relation_to_trip[r] += 1

        relation_hpt = {r: relation_to_trip[r] / len(relation_to_tail[r]) for h, r, t in self.all_hrt}
        relation_tph = {r: relation_to_trip[r] / len(relation_to_head[r]) for h, r, t in self.all_hrt}

        O_O_triple_hr_t, O_N_triple_hr_t, N_O_triple_hr_t, N_N_triple_hr_t = [], [], [], []

        for h, r, t in self.test_hrt:
            if relation_hpt[r] <= 1.5 and relation_tph[r] <= 1.5:
                O_O_triple_hr_t.append([h, r, t])
            elif relation_hpt[r] <= 1.5 and relation_tph[r] > 1.5:
                O_N_triple_hr_t.append([h, r, t])
            elif relation_hpt[r] > 1.5 and relation_tph[r] <= 1.5:
                N_O_triple_hr_t.append([h, r, t])
            elif relation_hpt[r] > 1.5 and relation_tph[r] > 1.5:
                N_N_triple_hr_t.append([h, r, t])

        O_O_hr_t_id = self.data_id(O_O_triple_hr_t)
        O_O_tr_h_id = [(i[2], i[1] + 1, i[0]) for i in O_O_hr_t_id]

        O_N_hr_t_id = self.data_id(O_N_triple_hr_t)
        O_N_tr_h_id = [(i[2], i[1] + 1, i[0]) for i in O_N_hr_t_id]

        N_O_hr_t_id = self.data_id(N_O_triple_hr_t)
        N_O_tr_h_id = [(i[2], i[1] + 1, i[0]) for i in N_O_hr_t_id]

        N_N_hr_t_id = self.data_id(N_N_triple_hr_t)
        N_N_tr_h_id = [(i[2], i[1] + 1, i[0]) for i in N_N_hr_t_id]

        return O_O_hr_t_id, O_N_hr_t_id, N_O_hr_t_id, N_N_hr_t_id, O_O_tr_h_id, O_N_tr_h_id, N_O_tr_h_id, N_N_tr_h_id


Data(data_dir="data/umls/", reverse=False)
