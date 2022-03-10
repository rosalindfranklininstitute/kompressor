import argparse
import os
import datetime
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import kompressor as kom
from glob import glob
import models
from callbacks.metrics import MetricsCallback
import compressors
import predictors


def decode_png_fn(path):
    return tf.io.decode_png(tf.io.read_file(path))


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def __option_parser():
    """
    Option parser for command line arguments.
    """
    parser = argparse.ArgumentParser(description="Setup training")
    parser.add_argument(
        "-l",
        "--levels",
        type=int,
        default=1,
        help="Select the number of resolutions to train the model over",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=1,
        help="Select neighbourhood size for prediction model",
    )
    parser.add_argument(
        "-k",
        "--kompressor",
        type=str,
        default="haiku",
        help="Select kompressor (haiku | )",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="cnn",
        help="Select model for kompressor (cnn | mlp)",
    )
    parser.add_argument(
        "-pr",
        "--predictor",
        type=str,
        default="regression",
        help="Select predictor algorithm for kompressor (regression | )",
    )
    parser.add_argument(
        "-log",
        "--logging",
        action='store_true',
        help="Turn logging on",
    )
    parser.add_argument(
        "-log_freq",
        "--logging_frequency",
        type=int,
        default=1000,
        help="Amount of epochs between logging",
    )
    parser.add_argument(
        "-df",
        "--data_folder",
        type=dir_path,
        help="Directory of data",
    )
    parser.add_argument(
        "-ts",
        "--test_size",
        type=int,
        default=1024,
        help="Size of the test dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100001,
        help="Number of training steps"
    )
    parser.add_argument(
        "-b_train",
        "--batch_train_size",
        type=int,
        default=5,
        help="Batch size for training"
    )

    parser.add_argument(
        "-b_test",
        "--batch_test_size",
        type=int,
        default=1,
        help="Batch size for testing"
    )
    parser.add_argument(
        "-chunk",
        "--chunk_size",
        type=int,
        default=65, #65
        help="Size of each chunk"
    )
    parser.add_argument(
        "-chunks_per_sample",
        "--chunks_per_sample",
        type=int,
        default=32, #32
        help="Amount of times to repeat the same data to allow for different random chunks to be sampled from it"
    )
    parser.add_argument(
        "-chunks_shuffle_buffer",
        "--chunks_shuffle_buffer",
        type=int,
        default=0, #512
        help="Buffer size for dataset shuffling"
    )
    args = parser.parse_args()
    return args


def main():
    """

    Returns
    -------
    """
    # Setup
    args = __option_parser()

    # Setup Args
    models.use_model(args.model)
    predictors.use_predictor(args.predictor)
    compressors.use_compressor(args.kompressor)
    dataset_paths = sorted(list(glob(args.data_folder + '*.png')))
    batch_train_size = args.batch_train_size
    batch_test_size = args.batch_test_size
    test_size = args.test_size
    epochs = args.epochs
    logging = args.logging
    dataset_split = len(dataset_paths) - test_size
    dataset_train = dataset_paths[:dataset_split]
    dataset_test = dataset_paths[dataset_split:]
    levels = args.levels
    padding = args.padding
    logging_frequency = args.logging_frequency
    chunk_size = args.chunk_size
    chunks_per_sample = args.chunks_per_sample
    chunks_shuffle_buffer = args.chunks_shuffle_buffer

    # Setup plugins
    model = models.get_model()
    predictor = predictors.get_predictor()
    compressor = compressors.get_compressor()

    print("dataset_train", len(dataset_train), "dataset_test", len(dataset_test))

    # Training
    ds_train = (
        tf.data.Dataset.from_tensor_slices(dataset_train)
        .repeat()
        .shuffle(len(dataset_train), reshuffle_each_iteration=True)
        .map(decode_png_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    ds_train = (
        kom.image.data.random_chunk_dataset(
            ds_train,
            padding=padding,
            chunk=chunk_size,
            chunks_per_sample=chunks_per_sample,
            chunks_shuffle_buffer=chunks_shuffle_buffer,
            levels=levels,
        )
        .batch(batch_train_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    ds_eval_test = (
        tf.data.Dataset.from_tensor_slices(dataset_test)
        .map(decode_png_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_test_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    encode_fn = kom.mapping.uint8.encode_values
    decode_fn = kom.mapping.uint8.decode_values

    compressor = compressor.Compressor(
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        padding=padding,
        model_fn=model.Model,
        predictions_fn=predictor.Predictor,
    ).init(ds_train)

    if logging:
        callbacks = [
            MetricsCallback(
                log_dir=os.path.join(
                    "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                ),
                chunk=None,
                ds_train=None,
                ds_test=ds_eval_test,
                log_freq=logging_frequency,
                levels=levels,
            )
        ]

        compressor.fit(
            ds_train=ds_train, start_step=0, end_step=epochs, callbacks=callbacks
        )
    else:
        compressor.fit(ds_train=ds_train, start_step=0, end_step=epochs)


if __name__ == "__main__":
    main()
