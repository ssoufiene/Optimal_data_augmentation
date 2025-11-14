import torch


class DefaultConfig:
    dataset_name = 'kitti'
    inner_model_class = 'unet'
    data_rater_model_class = 'DataRater'
    batch_size = 128
    train_split_ratio = 0.9
    inner_lr = 1e-3
    outer_lr = 1e-3
    meta_steps = 1000
    inner_steps = 2
    meta_refresh_steps = 10
    grad_clip_norm = 1.0
    num_inner_models = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_type = 'cross_entropy'
    save_data_rater_checkpoint = False
    log = False


class DataRaterConfig(DefaultConfig):
    pass