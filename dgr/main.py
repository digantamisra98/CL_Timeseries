#!/usr/bin/env python3
import argparse
import os.path
import numpy as np
import torch
import utils
from train import train
from dgr import Scholar
from models import Solver, Generator


parser = argparse.ArgumentParser("PyTorch implementation of Deep Generative Replay")

parser.add_argument("--experiment", type=str, choices=["bitcoin"])
parser.add_argument(
    "--replay-mode",
    type=str,
    default="generative-replay",
    choices=["exact-replay", "generative-replay", "none"],
)

parser.add_argument("--generator-iterations", type=int, default=3000)
parser.add_argument("--solver-iterations", type=int, default=1000)
parser.add_argument("--importance-of-new-task", type=float, default=0.3)
parser.add_argument("--lr", type=float, default=1e-04)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=1e-05)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--test-size", type=int, default=1024)
parser.add_argument("--sample-size", type=int, default=36)

parser.add_argument("--eval-log-interval", type=int, default=50)
parser.add_argument("--loss-log-interval", type=int, default=30)
parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
parser.add_argument("--no-gpus", action="store_false", dest="cuda")

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument("--train", action="store_true")
main_command.add_argument("--test", action="store_false", dest="train")


if __name__ == "__main__":
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda
    experiment = args.experiment
    capacity = args.batch_size * max(args.generator_iterations, args.solver_iterations)

    # setup bitcoin data using the following example
    # if experiment == 'bitcoin':
    #     d = pd.read_csv("/content/CL_Timeseries/cryp.csv")
    #     X = np.array(d["rate"].tolist())
    #     y= np.array(d["label"].tolist())
    #     X_train = X[:3500]
    #     y_train = y[:3500]
    #     X_test = X[3500:]
    #     y_test = y[3500:]
    #     train_data= TimeseriesDataset(X_train,y_train)
    #     train_present = X_train[3500:]
    #     test_data= TimeseriesDataset(X_test,y_test)
    #     # train_dataset = TimeseriesDataset(X_lstm, y_lstm, seq_len=4)
    #     train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = False)
    #     test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

    # # define the models.
    # solver = ## define the model from models.py
    solver = Solver(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
    )
    # generator = ## define the model from models.py
    generator = Generator(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        latent_dim=32,
    )

    label = "{experiment}-{replay_mode}-r{importance_of_new_task}".format(
        experiment=experiment,
        replay_mode=args.replay_mode,
        importance_of_new_task=(
            1 if args.replay_mode == "none" else args.importance_of_new_task
        ),
    )
    scholar = Scholar(label, generator=generator, solver=solver)

    # initialize the model.
    utils.gaussian_intiailize(scholar, std=0.02)

    # use cuda if needed
    if cuda:
        scholar.cuda()

    # determine whether we need to train the generator or not.
    train_generator = args.replay_mode == "generative-replay" or args.sample_log

    # run the experiment.
    if args.train:
        train(
            scholar,
            train_datasets,
            test_datasets,
            replay_mode=args.replay_mode,
            generator_iterations=(args.generator_iterations if train_generator else 0),
            solver_iterations=args.solver_iterations,
            importance_of_new_task=args.importance_of_new_task,
            batch_size=args.batch_size,
            test_size=args.test_size,
            sample_size=args.sample_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            loss_log_interval=args.loss_log_interval,
            eval_log_interval=args.eval_log_interval,
            checkpoint_dir=args.checkpoint_dir,
            collate_fn=utils.label_squeezing_collate_fn,
            cuda=cuda,
        )
