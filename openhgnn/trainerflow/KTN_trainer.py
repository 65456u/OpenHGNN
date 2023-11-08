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
from torch.utils.tensorboard import SummaryWriter


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
        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)
        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        self.source_type = args.source_type
        self.target_type = args.target_type
        self.labels = self.task.get_labels().to(self.device)
        self.source_type = args.source_type
        self.target_type = args.target_type
        self.dataset = self.task.dataset
        self.classifier=self.task.classifier
        self.task_type=args.task_type
        matching_w = {}
        self.label_dim=self.dataset.dims[self.task_type]
        # get target_type-source_type abbreviation, eg: 'paper' 'author' -> 'P-A'
        abbrev = self.target_type[0].upper() + "-" + self.source_type[0].upper()
        self.matching_path = self.dataset.meta_paths_dict[abbrev]
        for matching_id, relation in self.matching_path:
            matching_w[str(matching_id) + relation] = nn.Linear(
                self.args.out_dim, self.args.out_dim
            )
        self.matching_w = nn.ModuleDict(matching_w)
        for matching_id in self.matching_w.keys():
            nn.init.xavier_uniform_(self.matching_w[matching_id].weight)
        self.matching_loss = nn.MSELoss()

    def preprocess(self):
        pass

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            else:
                train_loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                modes = ["train", "valid"]
                if self.args.test_flag:
                    modes = modes + ["test"]

    def _full_train_step(self):


    def _mini_train_step(
        self,
    ):


    def get_matching_loss(self, edges, h_S, h_T):
        loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        if self.use_matching_loss == False:
            return loss
        h_Z = h_T
        for matching_id, edge in self.matching_path:
            h_Z = self.matching_w[str(matching_id) + edge](h_Z)
            h_Z = torch.spmm(g.adj_tensors(etype=edge, fmt="coo"), h_Z)
        loss = loss + self.matching_loss(h_S, h_Z)
        return loss
