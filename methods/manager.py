from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder
from .utils import Moment, dot_dist, batch_cos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from .utils import osdist
from .proto_sim import ProtoSimModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from transformers import BertModel, BertConfig, BertTokenizer


# 基于CRL的知识迁移，对这个文件和proto_sim.py两个文件修改
# 借鉴代码的地址：https://github.com/Alibaba-NLP/ProtoRE

class Manager(object):
    def __init__(self, args):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
        self.args = args


    # Use K-Means to select what samples to save, similar to at_least = 0








    def train(self, args):
        # set training batch

        for i in range(self.args.total_round):
            test_cur = []
            test_total = []
            # set random seed
            random.seed(self.args.seed + i * 100)

            # sampler setup
            sampler = data_sampler(args=self.args, seed=self.args.seed + i * 100)
            self.id2rel = sampler.id2rel
            self.rel2id = sampler.rel2id
            self.id2key = sampler.id2key
            self.key2id = sampler.id2key
            # encoder setup
            self.encoder = Encoder(args=self.args).to(self.args.device)

            # initialize memory and prototypes
            num_class = len(sampler.id2rel)
            memorized_samples = {}

            self.proto_sim_model = ProtoSimModel(self.args.class_num, self.args.feat_dim).to(self.args.device)  # 定义原形网络

            # load data and start computation

            history_relation_ids = []
            history_relation = []
            test_data_tasks = []
            relation_ids_tasks = []
            for steps, (
            training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(
                    sampler):

                print(current_relations)
                print("training_data: ", len(training_data))
                print("valid_data: ", len(valid_data))
                print("test_data: ", len(test_data))

                # Knowledge Transfer
                if steps > 0:
                    tokenizer = BertTokenizer.from_pretrained('./datasets/bert-base-uncased-vocab.txt')
                    bert = BertModel.from_pretrained('./datasets/bert-base-uncased')
                    bert_emb = bert.get_input_embeddings()
                    # bert.resize_token_embeddings(args.vocab_size + args.marker_size)
                    seen_relations_embeding = {}
                    for r in seen_relations:
                        token = tokenizer.encode(self.id2key[self.rel2id[r]])[1:-1]
                        token = torch.Tensor(token).long().unsqueeze(0)
                        emb_r = bert_emb(token).squeeze(0).mean(0)
                        seen_relations_embeding[self.rel2id[r]] = emb_r
                    for r in current_relations:
                        self.proto_sim_model.knowledge_get_proto(args, seen_relations_embeding, current_relations,
                                                            self.rel2id[r], seen_relations, self.rel2id)

                test_data_task = []
                init_train = []
                current_relation_ids=[]
                relation_ids_task = []
                cur_test_data = []
                for relation in current_relations:
                    history_relation.append(relation)
                    init_train += training_data[relation]
                    test_data_task += test_data[relation]
                    current_relation_ids.append(self.rel2id[relation])
                    history_relation_ids.append(self.rel2id[relation])
                    relation_ids_task.append(self.rel2id[relation])
                    cur_test_data += historic_test_data[relation]
                test_data_tasks.append(test_data_task)
                relation_ids_tasks.append(relation_ids_task)
                total_test_data = []
                for relation in seen_relations:
                    total_test_data += historic_test_data[relation]


                self.train_data(init_train, self.args.step1_epochs, current_relation_ids, is_mem=False)
                for relation in current_relations:
                    memorized_samples[relation] = self.select_data_(training_data[relation])
                if len(memorized_samples) > 0:
                    mem_train = []
                    for relation in history_relation:
                        mem_train += memorized_samples[relation]
                    self.train_data(mem_train, self.args.step2_epochs, history_relation_ids, is_mem=True, vat=False)
                total_test_data = []
                for relation in seen_relations:
                    total_test_data += historic_test_data[relation]
                cur_acc = self.evaluate_strict_model(cur_test_data, current_relation_ids)
                total_acc = self.evaluate_strict_model(total_test_data, history_relation_ids)


                print(f'Restart Num {i + 1}')
                print(f'task--{steps + 1}:')
                print(f'current test acc:{cur_acc}')
                print(f'history test acc:{total_acc}')
                test_cur.append(cur_acc)
                test_total.append(total_acc)
                print(test_cur)
                print(test_total)
                task_acc = []
                for i in range(len(test_data_tasks)):
                    acc = self.evaluate_strict_model(test_data_tasks[i], history_relation_ids)
                    task_acc.append(acc)
                print(f'Task Test Acc:{task_acc}')



    def get_mem_loss(self, N, rep, labels):
        pos=[]
        for idx in range(N):
            pos.append([])
            for idx2 in range(self.memlen):
                if labels[idx] == self.mem_labels[idx2]:
                    pos[idx].append(1)
                else:
                    pos[idx].append(0)
        p_batch = torch.from_numpy(np.array(pos)).to(self.args.device)
        n_batch = torch.ones(p_batch.size()).to(self.args.device) - p_batch
        sim = torch.mm(rep, self.mem_rep.t())

        sim_log_p = torch.nn.functional.logsigmoid(sim)
        sim_log_n = torch.nn.functional.logsigmoid(-sim)

        mem_loss = 0.0
        if p_batch.sum().item() > 0:
            mem_loss += -((sim_log_p * p_batch).sum()) / (p_batch.sum())
        if n_batch.sum() > 0:
            mem_loss += - ((sim_log_n * n_batch).sum()) / (n_batch.sum())
        return mem_loss

    def select_data_(self, sample_set):
        data_loader = get_data_loader(self.args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        self.encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature, rp = self.encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())
            # with torch.no_grad():
            #     rp = self.encoder.bert_forward(tokens)
            # features.append(rp.detach().cpu())

        features = np.concatenate(features)
        num_clusters = min(self.args.num_protos, len(sample_set))
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

        mem_set = []

        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
        return mem_set

    def _l2_normalize(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def train_data(self, train_data, epoch, relation_ids, is_mem):
        data_loader = get_data_loader(self.args, train_data, shuffle=True)
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 0.00001},
            {'params': self.proto_sim_model.parameters(), 'lr': 0.005}
        ])
        self.encoder.train()
        self.proto_sim_model.train()
        self.encoder.zero_grad()
        self.proto_sim_model.zero_grad()
        name = "memory" if is_mem else "init"
        for epoch_i in range(epoch):
            losses = []
            batch_losses = []
            proto_losses = []
            if is_mem: vat_losses = []
            td = tqdm(data_loader, desc=name)
            self.optimizer.zero_grad()
            for step, batch_data in enumerate(td):
                labels, tokens, ind = batch_data
                tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
                # adversarial training
                if is_mem:
                    self.encoder.eval()
                    with torch.no_grad():
                        _, reps = self.encoder.bert_forward(tokens)
                    reps_ = reps.clone().detach()
                    del reps
                    emb = self.encoder.encoder.encoder.embeddings
                    word_vector = emb(tokens).detach().cpu()
                    d = torch.rand(word_vector.shape).sub(0.5).to(self.args.device)
                    d = self.vec_l2_normalize(d)
                    d.requires_grad_()
                    word_vector_n = word_vector.to(self.args.device) + (d * 0.0001)
                    reps_n = self.encoder.get_rep_from_emb(tokens, word_vector_n)
                    vat_l = (torch.eye(labels.size()[0]).to(self.args.device) * torch.exp(-batch_cos(reps_, reps_n))).sum()
                    vat_l.backward(retain_graph=True)
                    e = d.grad.data.clone().detach()
                    d = self.vec_l2_normalize(e)
                    word_vector_n = word_vector.to(self.args.device) + (d * 0.0001)
                    self.encoder.train()
                    rep_n = self.encoder.get_rep_from_emb(tokens, word_vector_n)

                    vat_losses.append(vat_loss.item())
                    vat_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.proto_sim_model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                _, reps = self.encoder.bert_forward(tokens)
                batch_loss = self.batch_loss(reps, labels.to(self.args.device))
                proto_loss = self.proto_loss(reps, labels, relation_ids)
                loss = proto_loss + batch_loss
                losses.append(loss.item())
                batch_losses.append(batch_loss.item())
                proto_losses.append(proto_loss.item())
                td.set_postfix(loss=np.array(losses).mean())
                loss.backward()
                if not is_mem:
                    if (step + 1) % 1 == 0 or step == (len(data_loader) - 1):
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.proto_sim_model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.proto_sim_model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
            print(f"{name}_{epoch_i} loss is {np.array(losses).mean()}")
            if not is_mem:  print(f"{name}_{epoch_i} batch_loss is {np.array(batch_losses).mean()}")
            print(f"{name}_{epoch_i} proto_loss is {np.array(proto_losses).mean()}")
            if is_mem: print(f"{name}_{epoch_i} vat_loss is {np.array(vat_losses).mean()}")


    def vec_l2_normalize(self, d):
        for i in range(d.size()[0]):
            for j in range(d.size()[1]):
                d[i][j] = d[i][j]/(d[i][j]**2).sum()**.5
        return d

    def get_ave(self, train_data, relation_ids):
        data_loader = get_data_loader(self.args, train_data, shuffle=True)
        label_dict = {l: i for i, l in enumerate(relation_ids)}
        self.encoder.eval()
        avgs = []
        for i in range(len(relation_ids)):
            avgs.append([])
        td = tqdm(data_loader)
        for step, batch_data in enumerate(td):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
            _, reps = self.encoder.bert_forward(tokens)
            N = labels.size()[0]
            for i in range(N):
                avgs[label_dict[labels[i].item()]].append(reps[i].clone().detach())
        for l in label_dict.keys():
            avgs[label_dict[l]] = torch.stack(avgs[label_dict[l]]).mean(0)
            self.proto_sim_model.prototypes.weight.data[l] = avgs[label_dict[l]].to(self.args.device)
        avgs = torch.stack(avgs)
        # proto = self.proto_sim_model.prototypes(torch.tensor(relation_ids).long().to(self.args.device))
        # print(batch_cos(avgs, proto))
        return avgs


    def get_optimizer(self, args, encoder):
        print('Use {} optim!'.format(args.optim))

        def set_param(module, lr, decay=0):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ['undecay']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize

        params = set_param(encoder, args.learning_rate)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer


    def select_data(self, sample_set):
        data_loader = get_data_loader(self.args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        sims = []
        self.encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
            # with torch.no_grad():
            #     feature, rp = self.encoder.bert_forward(tokens)
            # features.append(feature.detach().cpu())
            with torch.no_grad():
                _, rep = self.encoder.bert_forward(tokens)
            sim = self.proto_sim_model(rep, labels.to(self.args.device))
            sims.append(sim.item())
        num = min(self.args.num_protos, len(sample_set))
        l = int(len(sim)/num)
        sims = torch.tensor(sims)
        _, sorted_id = torch.sort(sims, descending=True)
        mem_set = []
        for i in range(num):
            mem_set.append(sample_set[l*i])
        print(len(mem_set))
        return mem_set


    @torch.no_grad()
    def evaluate_strict_model(self, test_data, relation_ids):
        data_loader = get_data_loader(self.args, test_data, batch_size=1)
        self.encoder.eval()
        self.proto_sim_model.eval()
        n = len(test_data)
        label_dict = {l : i for i, l in enumerate(relation_ids)}
        relation_torch = torch.Tensor(relation_ids).long().to(self.args.device)
        correct = 0
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            label_id = label_dict[labels.item()]
            tokens = torch.stack([x.to(self.args.device) for x in tokens], dim=0)
            _, reps = self.encoder.bert_forward(tokens)
            proto_similarity = self.proto_sim_model(reps, relation_torch)

            predict_label = proto_similarity.max(1)[1].item()
            if predict_label == label_id:
                correct += 1
        return correct / n


    def batch_loss(self, reps, labels):
        p_batch = []
        N = labels.size()[0]
        for idx in range(N):
            p_batch.append([])
            for idx2 in range(N):
                if labels[idx] == labels[idx2] and idx != idx2:
                    p_batch[idx].append(1)
                else:
                    p_batch[idx].append(0)
        batch_similarity = batch_cos(reps, reps)
        p_batch = torch.from_numpy(np.array(p_batch)).to(self.args.device)
        b_sim_exp = torch.exp(batch_similarity / 0.07)
        batch_loss = torch.tensor(0.0).to(self.args.device)
        n_batch = torch.ones(p_batch.size()).cuda() - p_batch.cuda() - torch.eye(N).cuda()
        for i in range(N):
            if p_batch[i].sum().item() > 0:
                batch_loss += -torch.log((p_batch[i] * b_sim_exp[i]).sum() / ((p_batch + n_batch)[i] * b_sim_exp[i]).sum())
        return batch_loss

    def proto_loss(self, reps, labels, relation_ids):
        label_dict = {l: i for i, l in enumerate(relation_ids)}
        data_label_ids = list(map(label_dict.get, labels.numpy()))
        r_ids_torch = torch.Tensor(relation_ids).long().to(self.args.device)
        proto_similarity = self.proto_sim_model(reps, r_ids_torch)
        p_proto = np.eye(len(relation_ids), dtype=np.int8)[data_label_ids]
        p_proto = torch.from_numpy(p_proto).to(self.args.device)
        p_sim_exp = torch.exp(proto_similarity / 0.07)
        for i in range(len(data_label_ids)):
            proto_loss += -torch.log((p_proto[i] * p_sim_exp[i]).sum() / p_sim_exp[i].sum())
        return proto_loss











