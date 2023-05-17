# MIT License

# Copyright (c) 2023 Ao Li, Cong Feng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import pandas as pd
from GTSNE_MvAE.utils.metrics import (
    KMeans_Evaluate,
    mse_missing_part,
)
from GTSNE_MvAE.utils.torch_utils import convert_numpy, torch
from GTSNE_MvAE.vis.visualize import visualize_completion
from ..config import *
from .loss import *
from .model import *
from .postprocess import *
from .preprocess import *
from pathlib import Path as P
import random
from easydict import EasyDict
from pprint import pformat


class CompletionTrainer:
    def __init__(self, args: EasyDict) -> None:
        self.preprocess = PretrainPreprocess()
        self.postprocess = PretrainPostProcess()
        self.loss = MyLoss(args.lamda, args.before)
        self.history = []
        self.savedir = P(args.savedir)
        self.args = args
        self.model_path = self.savedir.joinpath("model.pth")

    def train(self):
        args = self.args
        inputs = self.preprocess(self.args, self.savedir)
        mm = inputs["mm"]

        model = GCN_IMC_Model(
            hidden_dims=args.hidden_dims,
            use_mlp=args.use_mlp,
            perplexity=args.perplexity,
            in_channels=inputs["data"].view_dims,
        ).to(args.device)
        optim = torch.optim.Adam(model.parameters(), lr=args["lr"])

        logging.info("************* Begin train completion model **************")
        best_mse = 999
        best_acc = -1
        x = inputs.copy()
        epoch = 0

        while True:
            model.train()
            x = model(x)
            loss = self.loss(x)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch += 1

            if (1 + epoch) % args.eval_epochs == 0:
                model.eval()
                with torch.no_grad():
                    x = model(x)
                mse = mse_missing_part(X_hat=x["X_hat"], X=x["X_gt"], M=~x["M"])
                cluster_metrics = KMeans_Evaluate(x["H_common"], x["data"])
                mm.update(MSE=mse, **cluster_metrics)
                logging.info(
                    f"epoch {epoch:04} loss {loss.item():.4f} {mm.report(current=True)}"
                )
                if mse < best_mse or cluster_metrics["ACC"] > best_acc:
                    best_mse = mse
                    # torch.save(model.encoder.state_dict(), self.model_path)
                    logging.info("Save best model to {}".format(self.model_path))
                    inputs.update(x)

                his = dict(loss=convert_numpy(loss), mse=mse, **cluster_metrics)
                self.history.append(his)

            if epoch >= args.epochs:
                logging.info("Train ends")
                break

        logging.info("************* Begin postprocessing **************")

        inputs["history"] = self.history
        self.postprocess(inputs, args)

    def train_minibatch(self):
        args = self.args
        dataset = MyDataset(args)
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            # drop_last=True,
        )
        test_dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
        )
        mm = dataset.mm

        model = GCN_IMC_Model(
            hidden_dims=args.hidden_dims,
            use_mlp=args.use_mlp,
            perplexity=args.perplexity,
            in_channels=dataset.data.view_dims,
        ).to(args.device)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        logging.info("************* Begin train completion model **************")
        best_mse = 999
        best_acc = -1
        inputs = {}

        for epoch in range(args.epochs):
            model.train()
            for x in train_dataloader:
                x = dataset.create_graphs(x)
                x = model(x)
                loss = self.loss(x)
                optim.zero_grad()
                loss.backward()
                optim.step()

            if (1 + epoch) % args.eval_epochs == 0:
                model.eval()
                with torch.no_grad():
                    for x in test_dataloader:
                        x = dataset.create_graphs(x)
                        x = model(x)
                        loss = self.loss(x)

                mse = mse_missing_part(X_hat=x["X_hat"], X=x["X_gt"], M=~x["M"])
                cluster_metrics = KMeans_Evaluate(x["H_common"], x["data"])
                mm.update(MSE=mse, **cluster_metrics)
                logging.info(
                    f"epoch {epoch:04} loss {loss.item():.4f} {mm.report(current=True)}"
                )
                if mse < best_mse or cluster_metrics["ACC"] > best_acc:
                    best_mse = mse
                    best_acc = cluster_metrics["ACC"]
                    # torch.save(model.encoder.state_dict(), self.model_path)
                    logging.info("Save best model to {}".format(self.model_path))
                    inputs.update(x)

                his = dict(loss=convert_numpy(loss), mse=mse, **cluster_metrics)
                self.history.append(his)

        logging.info("************* Begin postprocessing **************")
        inputs["history"] = self.history
        inputs["mm"] = mm
        self.postprocess(inputs, args)


def train(args: EasyDict):
    args.savedir = P(args.savedir)
    args.savedir.mkdir(parents=1, exist_ok=1)

    args.datapath = P(args.datadir).joinpath(args.dataname)
    if args.datapath.suffix != ".mat":
        args.datapath = args.datapath.with_suffix(".mat")

    assert args.datapath.exists(), args.datapath
    args.complete_ratio = 1 - args.incomplete_ratio

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True

    logfile = args.savedir.joinpath("train.log")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(logfile, mode="w", encoding="utf8"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
    )
    logging.info(f"Load args {pformat(args)}")

    trainer = CompletionTrainer(args)
    logging.info("Training begins")
    trainer.train()
    if args.save_vars:
        visualize_completion(trainer.savedir, args)


def train_on(
    dataname: str,
    incomplete_ratio: float,
    savedir: str,
    expname: str = None,
    perplexity: int = 10,
    lamda: float = 0.1,
    datadir: str = None,
    use_mlp: bool = False,
    epochs: int = 200,
    eval_epochs: int = 10,
    **kwds,
):
    args = load_config(CONF_DIR.joinpath(dataname).with_suffix(".yml"))

    if expname is None:
        expname = dataname
    savedir = P(savedir).joinpath(expname)

    if datadir is None:
        datadir = CONF_DIR.with_name("data")

    args_override = dict(
        dataname=dataname,
        incomplete_ratio=incomplete_ratio,
        datadir=datadir,
        perplexity=perplexity,
        lamda=lamda,
        use_mlp=use_mlp,
        savedir=savedir,
        epochs=epochs,
        eval_epochs=eval_epochs,
        **kwds,
    )
    args.update(args_override)
    train(args)


def collect_csv(
    indir: P,
    config_keys: List[str],
    outdir: P = None,
    metrics_keys: List[str] = None,
    title: str = None,
):
    def read_and_filter(file: P, keys: List[str]):
        data = json.loads(file.read_text())
        assert all(k in data for k in keys), (keys, data)
        if keys:
            return {k: v for k, v in data.items() if k in set(keys)}
        return data
    
    out = []
    for cfg_file in indir.rglob("args.json"):
        met_file = cfg_file.with_name("metrics.json")
        cfg = read_and_filter(cfg_file, config_keys)
        met = read_and_filter(met_file, metrics_keys)
        res = dict(**cfg, **met)
        out.append(res)

    df = pd.DataFrame.from_records(out)
    if not title:
        title = 'data'
    if outdir is None:
        outdir = indir
    file = outdir.joinpath(title).with_suffix(".csv")
    df.to_csv(file, index=False)
    logging.info(f"Save csv to {file}")
