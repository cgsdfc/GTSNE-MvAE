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
import math
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from GTSNE_MvAE.data.dataset import (
    MultiviewDataset,
    P,
    make_mask,
)
from GTSNE_MvAE.nn.ptsne_training import calculate_optimized_p_cond, make_joint
from GTSNE_MvAE.utils.metrics import MaxMetrics
from GTSNE_MvAE.utils.torch_utils import convert_tensor, nn, torch

from ..config import args


class MyDataset(Dataset):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.perplexity = config["perplexity"]
        outdir = config['savedir']

        data = MultiviewDataset(
            datapath=P(args.datapath),
            view_ids=args.views,
        )
        logging.info(
            f"Loaded dataset {data.name}, #views {data.viewNum} #samples {data.sampleNum}, complete_ratio {args.complete_ratio}"
        )

        M = make_mask(
            paired_rate=args.complete_ratio,
            sampleNum=data.sampleNum,
            viewNum=data.viewNum,
            kind="partial",
        )

        self.mm = MaxMetrics(MSE=False, ACC=True, NMI=True, PUR=True, F1=True, ARI=True)
        self.data = data
        self.M = M
        torch.save(M, outdir.joinpath("M.pt"))

        logging.info("Begin normalize X_view")
        scaler_view = [MinMaxScaler() for _ in range(data.viewNum)]
        X_view = [np.zeros_like(data.X[v]) for v in range(data.viewNum)]
        for v in range(data.viewNum):
            X_view[v][M[:, v]] = scaler_view[v].fit_transform(data.X[v][M[:, v]])

        logging.info("Begin normalize X_gt")
        X_gt = [None] * data.viewNum
        # scaler_view = [MinMaxScaler() for _ in range(data.viewNum)]
        for v in range(data.viewNum):
            X_gt[v] = scaler_view[v].fit_transform(data.X[v])

        self.X_view = X_view
        self.X_gt = X_gt

    def __len__(self):
        return self.data.sampleNum

    def __getitem__(self, index):
        data = self.data
        M = self.M[index, :]

        X_view = [self.X_view[v][index, :] for v in range(data.viewNum)]
        X_gt = [self.X_gt[v][index, :] for v in range(data.viewNum)]

        return dict(X_view=X_view, X_gt=X_gt, M=M)
    
    def create_graphs(self, inputs):
        X_view, X_gt, M = inputs["X_view"], inputs["X_gt"], inputs["M"]
        X_view = [X_view[v][M[:, v], :] for v in range(self.data.viewNum)]
        X_view = convert_tensor(X_view, torch.float, args.device)
        X_gt = convert_tensor(X_gt, torch.float, args.device)
        M = convert_tensor(M, torch.bool, args.device)
        perplexity = self.perplexity

        S_view = [
            calculate_optimized_p_cond(x, math.log2(perplexity), dev=args.device)
            for x in X_view
        ]

        P_view = [make_joint(s) for s in S_view]

        return dict(
            X_view=X_view,
            X_gt=X_gt,
            M=M,
            P_view=P_view,
            S_view=S_view,
            viewNum=self.data.viewNum,
            data=self.data,
        )

class PretrainPreprocess(nn.Module):
    """
    Preprocess of the completion pretraining stage:
    1. Construct an incomplete multi-view dataset.
    2. Construct the incomplete P and S.
    """

    def forward(self, args: dict, outdir):
        perplexity = args["perplexity"]

        data = MultiviewDataset(
            datapath=args.datapath,
            view_ids=args.views,
        )
        logging.info(
            "Loaded dataset {}, #views {} complete_ratio {}".format(
                data.name, data.viewNum, args.complete_ratio
            )
        )

        M = make_mask(
            paired_rate=args.complete_ratio,
            sampleNum=data.sampleNum,
            viewNum=data.viewNum,
            kind="partial",
        )

        X_view = [data.X[v][M[:, v]] for v in range(data.viewNum)]
        scaler_view = [MinMaxScaler() for _ in range(data.viewNum)]
        for v in range(data.viewNum):
            X_view[v] = scaler_view[v].fit_transform(X_view[v])
        X_view = convert_tensor(X_view, torch.float, args.device)

        X_gt = [None] * data.viewNum
        scaler_view = [MinMaxScaler() for _ in range(data.viewNum)]
        for v in range(data.viewNum):
            X_gt[v] = scaler_view[v].fit_transform(data.X[v])
        X_gt = convert_tensor(X_gt, torch.float, args.device)

        S_view = [
            calculate_optimized_p_cond(x, math.log2(perplexity), dev=args.device)
            for x in X_view
        ]

        P_view = [make_joint(s) for s in S_view]

        res = dict(
            data=data,
            viewNum=data.viewNum,
            M=convert_tensor(M, torch.bool, args.device),
            S_view=S_view,
            P_view=P_view,
            X_view=X_view,
            X_gt=X_gt,
            mm=MaxMetrics(MSE=False, ACC=True, NMI=True, PUR=True, F1=True, ARI=True),
        )
        if not args.save_vars:
            return res
        
        torch.save(S_view, outdir.joinpath("S_view.pt"))
        torch.save(P_view, outdir.joinpath("P_view.pt"))
        torch.save(X_view, outdir.joinpath("X_view.pt"))
        torch.save(M, outdir.joinpath("M.pt"))

        return res
