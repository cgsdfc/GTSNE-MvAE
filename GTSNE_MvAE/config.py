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

import argparse
from pathlib import Path as P
from typing import List, Dict, Any
import yaml
from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", dest="config_file", type=str, default="./conf.yml")


def load_config(config_file=None) -> EasyDict:
    if config_file is None:
        args = parser.parse_args()
        config_file = P(args.config_file)

    config_all: Dict[str, Dict[str, Any]] = yaml.load(config_file.open(), yaml.Loader)
    args = EasyDict(save_vars=True)
    for group in config_all.values():
        args.update(**group)
    return args

args = load_config()

CONF_DIR = P(__file__).parent.parent.joinpath('config')
assert CONF_DIR.is_dir(), CONF_DIR
