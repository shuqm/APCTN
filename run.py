#!/usr/bin/env python3

import argparse

from apctn.agents import APCTNAgent
from apctn.utils import check_pretrain_dir, load_json, process_config, set_default


def adjust_config(config):
    set_default(config, "validate_freq", value=1)
    set_default(config, "copy_checkpoint_freq", value=50)
    set_default(config, "debug", value=False)
    set_default(config, "cuda", value=True)
    set_default(config, "gpu_device", value=None)
    set_default(config, "pretrained_exp_dir", value=None)
    set_default(config, "continue_exp_dir", value=None)
    set_default(config, "agent", value="APCTNAgent")

    # data
    set_default(config.data_params, "aug_src", callback="aug")
    set_default(config.data_params, "aug_tgt", callback="aug")
    set_default(config.data_params, "num_workers", value=0)
    set_default(config.data_params, "image_size", value=224)
    set_default(config.data_params, "data_root", value="./data")
    set_default(config.data_params, "split_root", value="./data/splits")
    set_default(config.data_params, "train_val_split", value=False)

    # TBCN model defaults. These are implementation defaults and can be changed in JSON.
    set_default(config.model_params, "version", value="tbcn")
    set_default(config.model_params, "out_dim", value=512)
    set_default(config.model_params, "patch_size", value=16)
    set_default(config.model_params, "embed_dim", value=64)
    set_default(config.model_params, "transformer_depth", value=2)
    set_default(config.model_params, "num_heads", value=4)
    set_default(config.model_params, "mlp_ratio", value=2.0)
    set_default(config.model_params, "dropout", value=0.1)
    set_default(config.model_params, "prototype_threshold", value=0.3)
    set_default(config.model_params, "load_memory_bank", value=True)

    # losses
    num_loss = len(config.loss_params.loss)
    set_default(config.loss_params, "weight", value=[1] * num_loss)
    set_default(config.loss_params, "start", value=[0] * num_loss)
    set_default(config.loss_params, "end", value=[1000] * num_loss)
    if not isinstance(config.loss_params.temp, list):
        config.loss_params.temp = [config.loss_params.temp] * num_loss
    if len(config.loss_params.weight) != num_loss:
        raise ValueError("loss_params.weight and loss_params.loss must have the same length")
    set_default(config.loss_params, "m", value=0.5)
    set_default(config.loss_params, "T", value=0.05)
    set_default(config.loss_params, "pseudo", value=True)
    set_default(config.loss_params, "sample_ratio", value=None)
    set_default(config.loss_params, "k", value=1)

    # optimizer
    set_default(config.optim_params, "batch_size_src", callback="batch_size")
    set_default(config.optim_params, "batch_size_tgt", callback="batch_size")
    set_default(config.optim_params, "batch_size_lbd", callback="batch_size")
    set_default(config.optim_params, "momentum", value=0.9)
    set_default(config.optim_params, "nesterov", value=False)
    set_default(config.optim_params, "lr_decay_rate", value=0.1)
    set_default(config.optim_params, "lr_decay_schedule", value=None)
    set_default(config.optim_params, "cls_update", value=True)

    # clustering
    if config.loss_params.clus is not None:
        if config.loss_params.clus.type is None:
            config.loss_params.clus = None
        else:
            if not isinstance(config.loss_params.clus.type, list):
                config.loss_params.clus.type = [config.loss_params.clus.type]
            set_default(config.loss_params.clus, "tgt_GC", value=None)
            k = config.loss_params.clus.k
            n_k = config.loss_params.clus.n_k
            config.k_list = k * n_k
            config.loss_params.clus.n_kmeans = len(config.k_list)

    return config


def init_parser():
    parser = argparse.ArgumentParser(description="Train APCTN for limited-label UDA fault diagnosis")
    parser.add_argument(
        "--config",
        type=str,
        default="config/case1_shaft/T1_200rpm_to_250rpm.json",
        help="Path to an APCTN JSON configuration file.",
    )
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name under data/ and data/splits/")
    parser.add_argument("--source", type=str, default=None, help="Source domain name")
    parser.add_argument("--target", type=str, default=None, help="Target domain name")
    parser.add_argument("--num", type=str, default=None, help="Number of labeled source samples")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--seed", type=int, default=None, help="Random-seed override")
    return parser


def update_config(config_json, args):
    if args.dataset is not None:
        config_json["data_params"]["name"] = args.dataset
    if args.source is not None:
        config_json["data_params"]["source"] = args.source
    if args.target is not None:
        config_json["data_params"]["target"] = args.target
    if args.num is not None:
        config_json["data_params"]["fewshot"] = args.num
    if args.exp_id is not None:
        config_json["exp_id"] = args.exp_id
    elif args.source is not None and args.target is not None:
        config_json["exp_id"] = "{}_to_{}".format(args.source, args.target)
    if args.seed is not None:
        config_json["seed"] = args.seed
    if args.lr is not None:
        config_json["optim_params"]["learning_rate"] = args.lr


if __name__ == "__main__":
    args = init_parser().parse_args()
    config_json = load_json(args.config)
    update_config(config_json, args)

    pre_checkpoint_dir = check_pretrain_dir(config_json)
    config = adjust_config(process_config(config_json))

    agent = APCTNAgent(config)
    if pre_checkpoint_dir is not None:
        agent.load_checkpoint("model_best.pth.tar", pre_checkpoint_dir)

    try:
        agent.run()
        agent.finalise()
    except KeyboardInterrupt:
        pass
