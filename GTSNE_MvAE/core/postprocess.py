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

import json
import logging
from pathlib import Path as P
import pickle

from GTSNE_MvAE.utils.metrics import MaxMetrics
from GTSNE_MvAE.utils.torch_utils import (
    convert_numpy,
    nn,
    torch,
)

from ..config import args


def save_json(savedir: P, name: str, data: dict):
    logging.info("save_json {}".format(name))
    data = fix_json(data)
    json.dump(data, savedir.joinpath(name).open("w"), indent=4)


def save_pickle(savedir: P, name: str, data: dict):
    logging.info("save_pickle {}".format(name))
    data = convert_numpy(data)
    pickle.dump(data, savedir.joinpath(name).open("wb"))


def fix_json(data: dict):
    def func(x):
        if isinstance(x, P):
            return str(x)
        return x

    return {key: func(val) for key, val in data.items()}


class PretrainPostProcess(nn.Module):
    """
    Postprocess of pretraining stage.
    """

    def forward(self, inputs: dict, args: dict):
        savedir: P = args.savedir
        mm: MaxMetrics = inputs["mm"]
        metrics = mm.report(current=False)
        logging.info("After training, metrics {}".format(metrics))
        save_json(savedir, "metrics.json", metrics)
        save_json(savedir, "args.json", args)

        if not args.save_vars:
            return inputs
        
        torch.save(inputs["history"], savedir.joinpath("history.pt"))
        torch.save(inputs["H_common"], savedir.joinpath("H_common.pt"))
        torch.save(inputs["X_hat"], savedir.joinpath("X_hat.pt"))
        torch.save(inputs["X_gt"], savedir.joinpath("X_gt.pt"))

        return inputs
