# import os
# from dgl.data.utils import download, extract_archive
# from dgl.data import DGLDataset
# from dgl.data.utils import load_graphs


# class OAGDataset(DGLDataset):

#     _prefix = 'https://storage.cloud.google.com/graph_cs/'
#     _urls = {

#     }

#     def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
#         assert name in ['oag_cs']
#         self.data_path = './openhgnn/dataset/{}.bin.gz'.format(name)
#         self.g_path = './openhgnn/dataset/{}.bin'.format(name)
#         raw_dir = './openhgnn/dataset'
#         url = self._prefix + 'dataset/{}.bin.gz'.format(name)
#         super(OAGDataset, self).__init__(name=name,
#                                         url=url,
#                                         raw_dir=raw_dir,
#                                         force_reload=force_reload,
#                                         verbose=verbose)

#     def download(self):
#         # download raw data to local disk
#         # path to store the file
#         if os.path.exists(self.data_path):  # pragma: no cover
#            pass
#         else:
#             file_path = os.path.join(self.raw_dir)
#             # download file
#             download(self.url, path=file_path)
#         extract_archive(self.data_path, os.path.join(self.raw_dir, self.name))

#     def process(self):
#         # process raw data to graphs, labels, splitting masks
#         g, dims = load_graphs(self.g_path)
#         self._g = g[0]
#         self.dims=dims


#     def __getitem__(self, idx):
#         # get one example by index
#         assert idx == 0, "This dataset has only one graph"
#         return self._g

#     def __len__(self):
#         # number of data examples
#         return 1

#     def save(self):
#         # save processed data to directory `self.save_path`
#         pass

#     def load(self):
#         # load processed data from directory `self.save_path`
#         pass

#     def has_cache(self):
#         # check whether there are processed data in `self.save_path`
#         pass
import dgl
import dgl.function as fn
import torch as th
import numpy as np
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.data.utils import load_graphs, save_graphs
from . import load_acm_raw
from . import BaseDataset, register_dataset


@register_dataset("oag_dataset")
class OAGDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(OAGDataset, self).__init__(*args, **kwargs)
        self.g = None
        self.category = None
        self.num_classes = None
        self.has_feature = True
        print("Loading dataset", "oag_cs")
        self.load_graph_from_disk("./openhgnn/dataset/oag_cs.bin")
        self.meta_paths_dict = {
            "P-A": [("paper", "paper-author", "author")],
            "A-P": [("author", "author-paper", "paper")],
            "V-A": [
                ("venue", "venue-paper", "paper"),
                ("paper", "paper-author", "author"),
            ],
            "A-V": [
                ("author", "author-paper", "paper"),
                ("paper", "paper-venue", "venue"),
            ],
        }

    def load_graph_from_disk(self, file_path):
        glist, dims = dgl.load_graphs(file_path)
        self.g = glist[0]
        self.dims = dims

    def get_labels(self, task_type, node_type):
        assert task_type in ["L1", "L2"]
        return self.g.ndata[task_type][node_type]

    def get_split(self, node_type):
        train_mask = self.g.nodes[node_type].data["train_mask"]
        test_mask = self.g.nodes[node_type].data["test_mask"]
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
        valid_idx = train_idx
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.valid_idx = valid_idx
        return self.train_idx, self.valid_idx, self.test_idx
