"""
CRNN — Convolutional Recurrent Neural Network for handwriting recognition.

Architecture
------------
1.  CNN backbone  : series of Conv-BN-ReLU blocks with max-pooling.
                    Shrinks height to 1 while preserving a wide temporal dimension.
2.  Map-to-seq    : reshape feature map to a sequence of column vectors.
3.  BiLSTM stack  : two bidirectional LSTM layers model temporal context.
4.  Linear head   : project to (num_classes,) per time-step.
5.  Training loss : CTC (Connectionist Temporal Classification) — no alignment needed.

Input  : (B, 1, H, W)   grayscale word image, H=32, W=128
Output : (T, B, C)       log-softmax scores, T = W // 4 (time steps after pooling)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# CNN backbone
# ---------------------------------------------------------------------------

class CNNBackbone(nn.Module):
    """
    Input  (B, 1, 32, W)
    Output (B, 512, 1, W//4)

    Pool schedule:
        block1  pool (2,2) -> H=16, W/2
        block2  pool (2,2) -> H=8,  W/4
        block3  pool (2,1) -> H=4,  W/4
        block4  pool (4,1) -> H=1,  W/4   <- full height collapse
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            ConvBnRelu(1,   64),
            ConvBnRelu(64,  64),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 32->16, W->W/2
        )
        self.block2 = nn.Sequential(
            ConvBnRelu(64,  128),
            ConvBnRelu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 16->8, W/2->W/4
        )
        self.block3 = nn.Sequential(
            ConvBnRelu(128, 256),
            ConvBnRelu(256, 256),
            ConvBnRelu(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 8->4, W/4 unchanged
        )
        self.block4 = nn.Sequential(
            ConvBnRelu(256, 512),
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # 4->1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x   # (B, 512, 1, T)


# ---------------------------------------------------------------------------
# CRNN
# ---------------------------------------------------------------------------

class CRNN(nn.Module):
    """
    Full CRNN model.

    Args:
        num_classes : vocabulary size including CTC blank (config.NUM_CLASSES)
        rnn_hidden  : hidden units per direction in each BiLSTM layer
        rnn_layers  : number of stacked BiLSTM layers
        rnn_dropout : dropout between LSTM layers
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        rnn_hidden:  int = config.RNN_HIDDEN,
        rnn_layers:  int = config.RNN_LAYERS,
        rnn_dropout: float = config.RNN_DROPOUT,
    ):
        super().__init__()

        self.cnn = CNNBackbone()

        self.rnn = nn.LSTM(
            input_size  = 512,
            hidden_size = rnn_hidden,
            num_layers  = rnn_layers,
            dropout     = rnn_dropout if rnn_layers > 1 else 0.0,
            bidirectional = True,
            batch_first = False,   # expects (T, B, input)
        )

        self.head = nn.Linear(rnn_hidden * 2, num_classes)   # *2 for bidirectional

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, H, W)
        Returns:
            log_probs : (T, B, num_classes)  — log-softmax, ready for CTCLoss
        """
        # CNN
        feat = self.cnn(x)                  # (B, 512, 1, T)

        # Reshape to sequence
        B, C, H, T = feat.shape
        assert H == 1, f"Height after CNN should be 1, got {H}"
        feat = feat.squeeze(2)              # (B, 512, T)
        feat = feat.permute(2, 0, 1)        # (T, B, 512)

        # BiLSTM
        feat, _ = self.rnn(feat)            # (T, B, rnn_hidden*2)

        # Linear projection
        logits = self.head(feat)            # (T, B, num_classes)

        return F.log_softmax(logits, dim=2)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_greedy(self, x: torch.Tensor) -> List[str]:
        """Run greedy CTC decoding and return a list of predicted strings."""
        log_probs = self.forward(x)         # (T, B, C)
        preds = log_probs.argmax(dim=2)     # (T, B)
        preds = preds.permute(1, 0)         # (B, T)
        return [_greedy_decode(p.tolist()) for p in preds]


# ---------------------------------------------------------------------------
# Greedy CTC decoder  (used during training for quick metric snapshots)
# ---------------------------------------------------------------------------

def _greedy_decode(indices: list) -> str:
    """Collapse repeated tokens and remove blanks."""
    prev, chars = config.BLANK_IDX, []
    for idx in indices:
        if idx != config.BLANK_IDX and idx != prev:
            chars.append(config.IDX2CHAR.get(idx, ""))
        prev = idx
    return "".join(chars)


# Patch the missing List import at module level


# ---------------------------------------------------------------------------
# Model summary helper
# ---------------------------------------------------------------------------

def model_summary(model: nn.Module, input_size=(1, 1, 32, 128)):
    """Print parameter count and a brief layer summary."""
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nCRNN  |  Total params: {total:,}  |  Trainable: {train:,}")
    print("-" * 50)

    # forward pass for shape trace
    device = next(model.parameters()).device
    dummy  = torch.zeros(*input_size, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}   (T, B, num_classes)")
    print("-" * 50)


if __name__ == "__main__":
    net = CRNN().cuda()
    model_summary(net)
