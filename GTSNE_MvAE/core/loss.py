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

from GTSNE_MvAE.nn.ptsne_training import get_q_joint, loss_function
from GTSNE_MvAE.utils.torch_utils import Tensor, nn, F, torch, EPS_max


class MyLoss(nn.Module):
    """
    Manifold completion & alignment loss.
    """

    def __init__(self, lamda: float, before: bool):
        super().__init__()
        self.lamda = lamda
        self.loss_manifold_reg = ManifoldRegLossBeforeFusion() if before else ManifoldRegLoss()
        self.loss_completion = CompletionLoss()

    def forward(self, inputs: dict):
        return self.lamda * self.loss_manifold_reg(inputs) + self.loss_completion(
            inputs
        )


class ManifoldRegLossBeforeFusion(nn.Module):
    """
    t-SNE based manifold regularization loss.
    """
    def forward(self, inputs: dict):
        P_view: List[Tensor] = inputs["P_view"]
        H_view: Tensor = inputs["H_view"]
        M: Tensor = inputs["M"]
        viewNum: int = inputs["viewNum"]
        loss = 0

        for v in range(viewNum):
            h_view = H_view[v]
            q_view = get_q_joint(h_view)
            loss += loss_function(p_joint=P_view[v], q_joint=q_view)
        loss = loss / viewNum
        return loss


class ManifoldRegLoss(nn.Module):
    """
    t-SNE based manifold regularization loss.
    """
    def forward(self, inputs: dict):
        P_view: List[Tensor] = inputs["P_view"]
        H_common: Tensor = inputs["H_common"]
        M: Tensor = inputs["M"]
        viewNum: int = inputs["viewNum"]
        loss = 0

        for v in range(viewNum):
            h_common = H_common[M[:, v]]
            q_common = get_q_joint(h_common)
            loss += loss_function(p_joint=P_view[v], q_joint=q_common)
        loss = loss / viewNum
        return loss


class CompletionLoss(nn.Module):
    """
    MSE-based view completion loss
    """

    def forward(self, inputs: dict):
        X_hat: Tensor = inputs["X_hat"]
        M: Tensor = inputs["M"]
        X_view: Tensor = inputs["X_view"]
        loss = 0
        for v in range(inputs["data"].viewNum):
            loss += F.mse_loss(X_hat[v][M[:, v]], X_view[v])
        loss = loss / inputs["data"].viewNum
        return loss
