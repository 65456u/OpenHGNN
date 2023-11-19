import openhgnn
from openhgnn.trainerflow import KTNTrainer
from openhgnn import Experiment

experiment = Experiment(model='RGCN', dataset='acm4GTN', task='node_classification',gpu=0,mini_batch_flag=True)
experiment.run()

