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

from typing import List

from GTSNE_MvAE.nn.backbone import (
    GCN_Encoder_SDIMC,
    Imputer,
    NeuralMapper,
)
from GTSNE_MvAE.utils.torch_utils import (
    Tensor,
    nn,
    torch,
    F,
)
from ..config import args
from GTSNE_MvAE.utils.torch_utils import convert_tensor, nn, torch


class MultiviewEncoder(nn.Module):
    """
    The multi-view encoder part of the model.
    """

    def __init__(self, hidden_dims: int, use_mlp: bool, in_channels: List[int]):
        super().__init__()
        self.use_mlp = use_mlp
        self.hidden_dims = hidden_dims

        self.encoder_view = nn.ModuleList()
        for in_channel in in_channels:
            if self.use_mlp:
                encoder = NeuralMapper(dim_input=in_channel, dim_emb=self.hidden_dims)
            else:
                encoder = GCN_Encoder_SDIMC(
                    view_dim=in_channel, clusterNum=self.hidden_dims
                )

            self.encoder_view.append(encoder)

    def forward(self, inputs: dict):
        X_view: List[Tensor] = inputs["X_view"]
        M: Tensor = inputs["M"]
        S_view: List[Tensor] = inputs["S_view"]

        # Encoding
        sampleNum, viewNum = M.shape
        H_view = [None] * viewNum
        for v in range(viewNum):
            if self.use_mlp:
                h_tilde = self.encoder_view[v](X_view[v])
            else:
                h_tilde = self.encoder_view[v](X_view[v], S_view[v])
            H_view[v] = h_tilde

        # Fusion
        H_common = torch.zeros(sampleNum, self.hidden_dims).to(args.device)
        for v in range(viewNum):
            H_common[M[:, v]] += H_view[v]
        H_common = H_common / torch.sum(M, 1, keepdim=True)
        H_common = F.normalize(H_common)

        inputs["H_common"] = H_common
        inputs["H_view"] = H_view
        return inputs


class CompletionDecoder(nn.Module):
    """
    The decoder for view completion, i.e., completion-pretraining stage.
    """

    def __init__(self, hidden_dims: int, in_channels: List[int]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.decoder_view = nn.ModuleList()
        for v, in_channel in enumerate(in_channels):
            decoder = Imputer(in_channel, self.hidden_dims)
            self.decoder_view.append(decoder)

    def forward(self, inputs: dict):
        H_common: List[Tensor] = inputs["H_common"]
        X_hat = [None] * len(self.decoder_view)
        for v in range(len(self.decoder_view)):
            X_hat[v] = self.decoder_view[v](H_common)
        inputs["X_hat"] = X_hat
        return inputs


class GCN_IMC_Model(nn.Module):
    """
    The encoder-decoder model for representation learning & view completion.
    """

    def __init__(
        self,
        hidden_dims: int,
        use_mlp: bool,
        perplexity: int,
        in_channels: List[int],
    ):
        super().__init__()
        self.perplexity = perplexity
        self.viewNum = len(in_channels)
        self.encoder = MultiviewEncoder(
            hidden_dims=hidden_dims,
            use_mlp=use_mlp,
            in_channels=in_channels,
        )
        self.decoder = CompletionDecoder(hidden_dims, in_channels)

    def forward(self, inputs: dict):
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
        return inputs
