import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from . import BaseModel, register_model


@register_model("HMPNN")
class HMPNN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return HMPNN(
            args.in_dim, args.hid_dim, args.out_dim, hg.etypes, args.num_layers,args.device
        )

    def __init__(self, in_dim, hid_dim, out_dim, etypes, num_layers,device):
        super(HMPNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.etypes = etypes
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(HMPNNLayer(in_dim, hid_dim, etypes, activation=F.relu))
        self.device=device
        print("initing hmpnn model")
        for i in range(num_layers - 2):
            self.layers.append(HMPNNLayer(hid_dim, hid_dim, etypes, activation=F.relu))
        self.layers.append(HMPNNLayer(hid_dim, out_dim, etypes,activation=None))

    def forward(self, hg, h_dict):
        if hasattr(hg, "ntypes"):
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            i=0
            for layer, block in zip(self.layers, hg):
                print(i)
                i+=1
                block = block.to(self.device)
                # transfer h_dict to gpu
                for key in h_dict.keys():
                    h_dict[key] = h_dict[key].to(self.device)
                for k,v in h_dict.items():
                    print(v.device)
                print(block.device)
                h_dict = layer(block, h_dict)
                del block
                
        return h_dict


    def input_feature(self):
        return self.dataset.get_features()


class HMPNNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, etypes, activation=None):
        super(HMPNNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.etypes = etypes
        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(in_feat, out_feat, activation=activation)
                for rel in self.etypes
            },
            aggregate="sum",
        )

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        return h_dict
