import openhgnn
from openhgnn.trainerflow import KTNTrainer
from openhgnn import Experiment

experiment = Experiment(
    model="HMPNN",
    dataset="OAG_CS",
    task="ktn",
    gpu=0,
    lr=0.001,
    max_epoch=1000,
    num_layers=4,
    task_type="L1",
    source_type="paper",
    target_type="author",
    in_dim=1169,
    hid_dim=128,
    out_dim=128,
    rel_dim=128,
    batch_size=3072,
    use_matching_loss=True,
    matching_coeff=1,
    mini_batch_flag=False,
    evaluate_interval=10,
    ranking=True,
    source_train_batch=200,
    source_test_batch=50,
    target_test_batch=50,
    feature_name="feat",
    patience=10,
)
experiment.run()
