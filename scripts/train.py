import argparse
import os
from datetime import datetime

import kompressor as kom
import orbax.checkpoint as ocp
import yaml
from sklearn import model_selection
from torchvision.transforms import transforms

import wandb


def main():
    parser = argparse.ArgumentParser(
        prog="Kompressor Trainer", description="Trains a Kompressor model"
    )

    parser.add_argument(
        "-i",
        dest="filename",
        required=True,
        help="input config for training",
        metavar="FILE",
        type=lambda x: kom.config.parser.is_valid_file(parser, x),
    )
    args = parser.parse_args()
    config = kom.config.parser.get_config(args.filename)
    print(config)
    if not os.path.exists(config["train"]["dataset_path"]):
        raise FileNotFoundError(
            f"Dataset at {config['train']['dataset_path']} Not Found"
        )
    dataset_name = config["train"]["dataset_path"].split("/")[-1]
    config["train"]["dataset_name"] = dataset_name
    data_paths_and_frames = kom.dataset.mrc_dataset.get_data_paths_and_frames(
        [config["train"]["dataset_path"]]
    )
    train_data, _ = model_selection.train_test_split(
        data_paths_and_frames,
        test_size=config["general"]["dataset_split"],
        random_state=config["general"]["dataset_seed"],
    )
    data_transforms = transforms.Compose(
        [kom.transforms.transform.RandomChunkDataset(config["train"]["padding"])]
    )
    train_dataset = kom.dataset.mrc_dataset.MRCFileDataset(train_data, data_transforms)

    train_dataloader = kom.dataloader.np.NumpyLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    log_freq = len(train_dataloader)
    checkpoint_path = config["train"]["checkpoint_directory"]
    checkpoint_path = os.path.join(
        checkpoint_path, f"ckpt_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )
    os.makedirs(checkpoint_path, exist_ok=True)
    options = ocp.CheckpointManagerOptions()
    mngr = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(checkpoint_path),
        options=options,
    )

    encode_fn = kom.mapping.uint16.encode_values
    decode_fn = kom.mapping.uint16.decode_values
    model_class_ = getattr(kom.models, config["model"]["name"])
    model = model_class_(**config["model"]["attributes"])
    compressor = kom.compressors.FlaxKompressor(
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        padding=config["train"]["padding"],
        model_fn=model,
        predictions_fn=kom.predictors.regression.regression_predictor,
    ).init(train_dataloader, seed=config["train"]["seed"])
    callbacks = [
        kom.callbacks.MetricsCallback(
            chunk=None,
            ds_train=None,
            ds_test=None,
            log_freq=log_freq,
            levels=config["train"]["levels"],
        )
    ]

    for key, value in vars(model).items():
        if not key.startswith("_"):
            config["model"]["attributes"][key] = value
    train_config_path = os.path.join(checkpoint_path, "config.yaml")
    with open(train_config_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    #
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project=config["project_name"],
        # track hyperparameters and run metadata
        config=config,
    )

    compressor.fit(
        ds_train=train_dataloader,
        start_step=0,
        end_step=config["train"]["epochs"],
        learning_rate=config["train"]["learning_rate"],
        checkpoint_manager=mngr,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
