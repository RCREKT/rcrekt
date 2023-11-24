import torch
from torch import nn
import torch.nn.functional as F
import random
from .utils import batch_cos


class ProtoSimModel(nn.Module):

    def __init__(self, relation_count, embedding_width):
        nn.Module.__init__(self)
        self.relation_count = relation_count
        self.prototypes = nn.Embedding(relation_count, embedding_width)

    def forward(self, hidden, rel_ids):
        protos = self.prototypes(rel_ids)
        # p_sim = torch.mm(hidden, protos.t())
        p_sim = batch_cos(hidden, protos)
        return p_sim




    def get_distance(self, embeding1, embeding2):
        return -1 * (F.cosine_similarity(embeding1.squeeze(0), embeding2.squeeze(0), dim=0) - 1)

    def knowledge_get_proto(self, args, rel_emb, current_relations, cur_rel_id, seen_relations, rel2id):
        sims = {}
        sum = 0
        for rel in seen_relations:
            if rel not in current_relations:
                sims[rel2id[rel]] = F.cosine_similarity(rel_emb[rel2id[rel]], rel_emb[cur_rel_id], dim=0).item()
                if sims[rel2id[rel]] < 0.7:
                    sims[rel2id[rel]] = 0.0
                sum += sims[rel2id[rel]]
        # proto = self.prototypes.weight.data[cur_rel_id].to(args.device) * 0.0000001

        if sum > 0:
            # proto = torch.zeros(self.prototypes.weight.data[cur_rel_id].size()).to(args.device)
            proto = self.prototypes.weight.data[cur_rel_id].clone() * .5
            print("###########################################")
            # print(sum)
            # print(proto)
            for key in sims.keys():
                similar = sims[key] / sum
                # print(similar)
                pro = self.prototypes.weight.data[key].clone().to(args.device)
                # print(pro)
                proto += similar * pro
            # print(proto)
            self.prototypes.weight.data[cur_rel_id] = proto

    def get_proto(self, relation_id):
        return self.prototypes.weight.data[relation_id]

    # def forward(self, relation_embedding, relation_id):
    #
    #     protos = self.prototypes(relation_id)
    #     similarity = 1 - 1 / (1 + torch.exp(
    #         (torch.sum(protos.unsqueeze(0) * relation_embedding.unsqueeze(0), 1) - 384) / 100))  # scale the value to avoid gradient explosion
    #     predict_relation = self.classification_layer(protos)
    #
    #     return similarity.cuda(), predict_relation.cuda()
