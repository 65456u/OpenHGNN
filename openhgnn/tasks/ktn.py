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
        self.classifier = Classifier(args.out_dim, self.dataset.dims[self.task_type])

    def get_graph(self):
        return self.dataset.g

    def get_split(self, node_type):
        return self.dataset.get_split(node_type)

    def get_labels(self):
        return self.dataset.g.ndata[self.task_type]

    def get_loss(self, y_pred, y_true):
        return self.classifier.calc_loss(y_pred, y_true)

    def evaluate(self, y_pred, y_true):
        return self.classifier.calc_acc(y_pred, y_true)

    def get_loss_fn(self):
        return self.classifier.calc_loss


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.0


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1.0 / (r[0] + 1) if r.size else 0.0 for r in rs]


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
            test_res = []
            test_ndcg = []
            for ai, bi in zip(y_true, torch.argsort(y_pred, dim=-1, descending=True)):
                resi = ai[bi].cpu().numpy()
                test_res += [resi]
                test_ndcg += [ndcg_at_k(resi, len(resi))]
            test_ndcg = np.average(test_ndcg)
            test_mrr = np.average(mean_reciprocal_rank(test_res))
            return test_ndcg, test_mrr
        else:
            y_pred = torch.argmax(y_pred, dim=1).cpu()
            y_true = torch.argmax(y_true, dim=1).cpu()
            return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(
                y_true, y_pred, average="macro"
            )
            