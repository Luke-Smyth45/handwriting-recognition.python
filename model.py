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
    """Single Conv → BatchNorm → ReLU building block used throughout the CNN."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        # bias=False because BatchNorm already handles the bias term
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),   # normalises activations for stable training
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

        # Each block doubles the number of channels and shrinks the height
        self.block1 = nn.Sequential(
            ConvBnRelu(1,   64),
            ConvBnRelu(64,  64),
            nn.MaxPool2d(kernel_size=2, stride=2),      # height 32→16, width halved
        )
        self.block2 = nn.Sequential(
            ConvBnRelu(64,  128),
            ConvBnRelu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),      # height 16→8, width halved again
        )
        self.block3 = nn.Sequential(
            ConvBnRelu(128, 256),
            ConvBnRelu(256, 256),
            ConvBnRelu(256, 256),
            # Pool only vertically — keep full width so we don't lose horizontal detail
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # height 8→4, width unchanged
        )
        self.block4 = nn.Sequential(
            ConvBnRelu(256, 512),
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512),
            # Final vertical collapse — height becomes 1, turning the image into a sequence
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # height 4→1
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
        # Step 1: CNN extracts visual features, collapses height to 1
        feat = self.cnn(x)                  # (B, 512, 1, T)

        # Step 2: Reshape the 2D feature map into a 1D sequence of column vectors
        B, C, H, T = feat.shape
        assert H == 1, f"Height after CNN should be 1, got {H}"
        feat = feat.squeeze(2)              # remove the H=1 dimension → (B, 512, T)
        feat = feat.permute(2, 0, 1)        # reorder for LSTM: (T, B, 512)

        # Step 3: BiLSTM reads the sequence and adds temporal context
        feat, _ = self.rnn(feat)            # (T, B, rnn_hidden*2)

        # Step 4: Project each time step to a probability over the vocabulary
        logits = self.head(feat)            # (T, B, num_classes)

        # log_softmax gives log-probabilities — required by CTCLoss
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
    """
    CTC greedy decoding: remove blank tokens and collapse consecutive repeated characters.
    e.g. [h, h, blank, e, l, l, blank, l, o] → "hello"
    """
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
