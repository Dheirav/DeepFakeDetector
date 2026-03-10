import torch
import torch.nn as nn


class FFTLayer(nn.Module):
    """
    Produces a single FFT magnitude channel from the input tensor.

    Input:
        B x 3 x H x W

    Output:
        B x 1 x H x W  (magnitude only)
    """

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)

        fft = torch.fft.fft2(gray)
        fft = torch.fft.fftshift(fft)

        magnitude = torch.log(torch.abs(fft) + 1e-8)

        magnitude = (magnitude - magnitude.min()) / (
            magnitude.max() - magnitude.min() + 1e-8
        )

        return magnitude