import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator
import torch
import numpy as np
from sklearn import metrics


@register_task("ktn")
class KTN(BaseTask):
    def __init__(self, args):
        super(KTN, self).__init__()
        print("starting ktn task")
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, "ktn", logger=self.logger)
        self.g = self.dataset.g
        self.args = args
        self.task_type = args.task_type
        self.ranking = args.ranking
        self.classifier = Classifier(
            args.out_dim, self.dataset.dims[self.task_type], self.ranking
        )

    def get_graph(self):
        return self.dataset.g

    def get_split(self, node_type, device="cpu"):
        return self.dataset.get_split(node_type, device=device)

    def get_labels(self):
        return self.dataset.g.ndata[self.task_type]

    def get_loss(self, y_pred, y_true):
        return self.classifier.calc_loss(y_pred, y_true)

    def evaluate(self, y_pred, y_true):
        return self.classifier.calc_acc(y_pred, y_true)

    def get_loss_fn(self):
        return self.classifier.calc_loss


def _mrr(indices, true_labels):
    if true_labels.dim() == 1:
        true_labels = true_labels.unsqueeze(0)
    true_indices = true_labels.argmax(dim=1, keepdim=True)
    ranks = (indices == true_indices).nonzero(as_tuple=True)[1] + 1
    reciprocal_ranks = 1.0 / ranks.float()
    has_relevant = (true_labels.max(dim=1).values == 0).nonzero(as_tuple=True)[0]
    reciprocal_ranks[has_relevant] = 0
    mrr = reciprocal_ranks.mean()
    return mrr.item()


def _ndcg(indices, true_relevance):
    k = true_relevance.shape[1]
    sorted_true_relevance = torch.gather(true_relevance, 1, indices)
    discounts = torch.log2(torch.arange(k, device=true_relevance.device).float() + 2.0)
    dcg = (sorted_true_relevance[:, :k] / discounts).sum(dim=1)
    true_indices = true_relevance.argsort(descending=True, dim=1)
    ideal_sorted_relevance = torch.gather(true_relevance, 1, true_indices)
    idcg = (ideal_sorted_relevance[:, :k] / discounts).sum(dim=1)
    idcg[idcg == 0] = 1
    ndcg = dcg / idcg

    return ndcg.mean().item()


class Classifier(nn.Module):
    def __init__(self, n_in, n_out, ranking=False):
        super(Classifier, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.ranking = ranking
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.criterion = torch.nn.KLDivLoss(reduction="batchmean")
        nn.init.xavier_uniform_(self.linear.weight)

    def get_parameters(self):
        ml = list()
        ml.append({"params": self.linear.parameters()})
        return ml

    def forward(self, x):
        y = self.linear(x)
        return torch.log_softmax(y, dim=-1)

    def calc_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)

    def calc_acc(self, y_pred, y_true):
        if self.ranking:
            indices = y_pred.argsort(descending=True, dim=1)
            mrr = _mrr(indices, y_true)
            ndcg = _ndcg(indices, y_true)
            return ndcg, mrr
        else:
            y_pred = torch.argmax(y_pred, dim=1).cpu()
            y_true = torch.argmax(y_true, dim=1).cpu()
            return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(
                y_true, y_pred, average="macro"
            )
