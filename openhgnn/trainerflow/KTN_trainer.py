import dgl
import torch
from tqdm import tqdm
from ..utils.sampler import get_node_data_loader
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping, to_hetero_idx, to_homo_feature, to_homo_idx
from ..tasks import build_task
import warnings
import torch.nn as nn
import dgl.sparse as dglsp
from torch.utils.tensorboard import SummaryWriter


def process_category(labels: torch.tensor, num_classes: int) -> torch.tensor:
    if labels.shape[1] != num_classes:
        processed_labels = torch.zeros((labels.shape[0], num_classes))
        num_indices = torch.count_nonzero(labels + 1, dim=1)
        for i in range(labels.shape[0]):
            indices = labels[i, : num_indices[i]].to(torch.int)
            processed_labels[i, indices] = 1 / num_indices[i].clamp(min=1)
    else:
        processed_labels = labels
    return processed_labels


@register_flow("KTN_trainer")
class KTNTrainer(BaseFlow):
    """KTNtrainer flows.
    Supported Model: HMPNN
    Supported Dataset: oag_cs

    """

    def __init__(self, args):
        super(KTNTrainer, self).__init__(args)
        self.args = args
        self.model_name = args.model
        self.device = args.device
        # self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)
        self.model = (
            build_model(self.model)
            .build_model_from_args(self.args, self.hg)
            .to(self.device)
        )
        self.source_type = args.source_type
        self.target_type = args.target_type
        self.source_type = args.source_type
        self.target_type = args.target_type
        self.dataset = self.task.dataset
        self.use_matching_loss = args.use_matching_loss
        self.classifier = self.task.classifier
        self.task_type = args.task_type
        self.mini_batch_flag = args.mini_batch_flag
        self.num_layers = args.num_layers
        self.g = self.dataset.g
        self.batch_size = args.batch_size
        (
            self.source_train_idx,
            self.source_val_idx,
            self.source_test_idx,
        ) = self.task.get_split(self.source_type)
        (
            self.target_train_idx,
            self.target_val_idx,
            self.target_test_idx,
        ) = self.task.get_split(self.target_type)
        self.source_labels = self.dataset.get_labels(self.task_type, self.source_type)
        self.target_labels = self.dataset.get_labels(self.task_type, self.target_type)
        self.matching_coeff = args.matching_coeff
        matching_w = {}
        self.label_dim = self.dataset.dims[self.task_type]
        # get target_type-source_type abbreviation, eg: 'paper' 'author' -> 'P-A'
        abbrev = self.target_type[0].upper() + "-" + self.source_type[0].upper()
        self.matching_path = self.dataset.meta_paths_dict[abbrev]
        for matching_id, relation in enumerate(self.matching_path):
            matching_w[str(matching_id) + relation[1]] = nn.Linear(
                self.args.out_dim, self.args.out_dim
            )
        self.matching_w = nn.ModuleDict(matching_w)
        for matching_id in self.matching_w.keys():
            nn.init.xavier_uniform_(self.matching_w[matching_id].weight)
        self.matching_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.matching_w.parameters()},
                {"params": self.classifier.parameters()},
            ],
            lr=self.args.lr,
        )
        self.writer = SummaryWriter(f"./openhgnn/output/{self.model_name}/")

    def preprocess(self):
        pass

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience)
        epoch_iter = tqdm(range(self.max_epoch))
        if not self.args.mini_batch_flag:
            self.train_labels = process_category(
                self.source_labels[self.source_train_idx], self.label_dim
            )
            self.source_test_labels = process_category(
                self.source_labels[self.source_test_idx], self.label_dim
            )
            self.target_test_labels = process_category(
                self.target_labels[self.target_test_idx], self.label_dim
            )
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                self.logger.info("Epoch {:d} | mini-batch training".format(epoch))
                train_loss = self._mini_train_step()
                self.logger.info(
                    "Epoch {:d} | Train Loss {:.4f}".format(epoch, train_loss)
                )
                # matching loss
                h_dict = self.g.ndata["feat"].to(self.device)
                h_dict = self.model(self.g, h_dict)
                h_S = h_dict[self.source_type]
                h_T = h_dict[self.target_type]
                matching_loss = self.get_matching_loss(h_S, h_T)
                print(matching_loss)
                loss = train_loss + self.matching_coeff * matching_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                early_stop = stopper.loss_step(loss, self.model)
                if early_stop:
                    self.logger.train_info("Early Stop!\tEpoch:" + str(epoch))
                    break
            else:
                self.logger.info("Epoch {:d} | full-batch training".format(epoch))
                train_loss = self._full_train_step()
                self.logger.info(
                    "Epoch {:d} | Train Loss {:.4f}".format(epoch, train_loss)
                )
                h_dict = self.g.ndata["feat"]
                h_dict = self.model(self.g, h_dict)
                h_S = h_dict[self.source_type]
                h_T = h_dict[self.target_type]
                matching_loss = self.get_matching_loss(h_S, h_T)
                print(matching_loss)
                loss = train_loss + self.matching_coeff * matching_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if epoch % self.evaluate_interval == 0:
                    self.logger.info("Start evaling")
                    acc = self._full_test_step()
                    print(acc)
                    # use summary writer to record the acc

    def _full_train_step(self):
        self.model.train()
        self.matching_w.train()
        self.classifier.train()
        self.g = self.g.to(self.device)
        h_dict = self.g.ndata["feat"]
        logits = self.model(self.g, h_dict)[self.source_type][self.source_train_idx]
        pred_y = self.classifier(logits)
        loss = self.loss_fn(pred_y, self.train_labels)
        return loss

    def _mini_train_step(
        self,
    ):
        self.model.train()
        self.matching_w.train()
        self.classifier.train()
        loader_tqdm = tqdm(self.train_loader, ncols=120)
        all_loss = 0
        for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
            h = {}
            for ntype in input_nodes.keys():
                h[ntype] = self.g.ndata["feat"][ntype][input_nodes[ntype]].to(
                    self.device
                )
            lbl = self.source_labels[seeds[self.source_type]]
            # normalize labels
            lbl = process_category(lbl, self.label_dim)
            print(lbl.device)
            print("Current allocated memory:", torch.cuda.memory_allocated())
            h = self.model(blocks, h)
            logits = self.classifier(h[self.source_type])
            loss = self.loss_fn(logits, lbl)
            self.logger.info("loss: {:.4f}".format(loss))
            all_loss += loss
            break
        return all_loss

    def get_matching_loss(self, h_S, h_T):
        loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        if self.use_matching_loss == False:
            return loss
        h_Z = h_T
        for matching_id, edge in enumerate(self.matching_path):
            h_Z = self.matching_w[str(matching_id) + edge[1]](h_Z)
            adj = self.g.adj(etype=edge).transpose()
            h_Z = dglsp.spmm(adj, h_Z)
            print(h_Z.shape)
        loss = loss + self.matching_loss(h_S, h_Z)
        return loss

    def preprocess(self):
        if self.args.mini_batch_flag:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
            self.train_loader = dgl.dataloading.DataLoader(
                self.g,
                {self.source_type: self.source_train_idx},
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                # num_workers=4,
            )
        return

    def _full_test_step(self):
        self.model.eval()
        self.classifier.eval()
        self.matching_w.eval()
        with torch.no_grad():
            h_dict = self.g.ndata["feat"]
            logits = self.model(self.g, h_dict)
            source_y = self.classifier(logits[self.source_type])[self.source_test_idx]
            origin_target_y = self.classifier(logits[self.target_type])[
                self.target_test_idx
            ]
            ktn_logits = logits[self.target_type]
            for matching_id, edge in enumerate(self.matching_path):
                ktn_logits = self.matching_w[str(matching_id) + edge[1]](ktn_logits)
            ktn_target_y = self.classifier(ktn_logits)[self.target_test_idx]
            source_acc = self.task.evaluate(source_y, self.source_test_labels)
            target_acc = self.task.evaluate(origin_target_y, self.target_test_labels)
            ktn_acc = self.task.evaluate(ktn_target_y, self.target_test_labels)
            return source_acc, target_acc, ktn_acc
