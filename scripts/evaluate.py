import argparse
import os
from collections import defaultdict
import jax.numpy as jnp
from tqdm import tqdm

import kompressor as kom
import orbax.checkpoint as ocp
from sklearn import model_selection

import wandb


def evaluate_compressor(dataset, levels, compressor, mode, epoch):
    summaries = defaultdict(list)
    for highres in tqdm(dataset):
        lowres, level_encoded_maps = compressor.encode(
            highres, levels=levels, chunk=None, debug=True
        )

        for level, (
            level_highres,
            (level_encoded_maps, _),
            level_lowres,
        ) in enumerate(level_encoded_maps):
            level_highres_maps = kom.image.maps_from_highres(level_highres)

            writer_path = os.path.join(
                f"{mode}/level={level}/",
                f'lowres={"x".join(map(str, level_lowres.shape[1:]))}',
            )

            for label in level_encoded_maps.keys() & level_highres_maps.keys():
                summaries[f"{writer_path}/{label}/run_length"].extend(
                    list(
                        kom.image.metrics.mean_run_length(
                            level_encoded_maps[label]
                        ).flatten()
                    )
                )

    for key in summaries.keys():
        wandb.log({key: (jnp.array(summaries[key]).flatten().mean())}, step=epoch)


def main():
    parser = argparse.ArgumentParser(
        prog="Kompressor Evaluate", description="Evaluate Kompressor models"
    )

    parser.add_argument(
        "-i",
        dest="filename",
        required=True,
        help="Checkpoint folder. Expects to contain a config.yaml file",
        metavar="FILE",
        type=lambda x: kom.config.parser.is_valid_file(parser, x),
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["train", "test", "whole"],
        required=True,
        help="Dataset to check compression performance on",
    )
    args = parser.parse_args()
    config_file = os.path.join(args.filename, "config.yaml")
    config = kom.config.parser.get_config(config_file)
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
    if args.mode == "train":
        eval_data, _ = model_selection.train_test_split(
            data_paths_and_frames,
            test_size=config["general"]["dataset_split"],
            random_state=config["general"]["dataset_seed"],
        )
    elif args.mode == "test":
        _, eval_data = model_selection.train_test_split(
            data_paths_and_frames,
            test_size=config["general"]["dataset_split"],
            random_state=config["general"]["dataset_seed"],
        )
    elif args.mode == "whole":
        eval_data = data_paths_and_frames
    eval_dataset = kom.dataset.mrc_dataset.MRCFileDataset(eval_data)
    eval_dataloader = kom.dataloader.np.NumpyLoader(
        eval_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
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
    )
    options = ocp.CheckpointManagerOptions()
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    mngr = ocp.CheckpointManager(args.filename, orbax_checkpointer, options=options)
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project=config["project_name"],
        # track hyperparameters and run metadata
        config=config,
    )
    for checkpoint in tqdm(range(config["train"]["epochs"])):
        compressor.avg_params = mngr.restore(f"{args.filename}/{checkpoint}")["model"][
            "params"
        ]
        evaluate_compressor(
            eval_dataloader,
            config["train"]["levels"],
            compressor,
            args.mode,
            checkpoint,
        )


if __name__ == "__main__":
    main()
