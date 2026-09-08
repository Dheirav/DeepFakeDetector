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
        # dim must be given. torch.fft.fftshift defaults to shifting EVERY axis,
        # which includes the batch: with N samples it rolls the batch by N//2, so
        # sample i received sample (i + N//2)'s spectrum. Only the two spatial
        # axes should be centred.
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        magnitude = torch.log(torch.abs(fft) + 1e-8)

        # Normalise per sample. Bare .min()/.max() reduce over EVERY dimension
        # including the batch, so an image's FFT channel depended on whichever
        # other images shared its batch: the same picture's channel mean shifted
        # by 0.136 -- 14% of its range -- between batch_size=64 training and
        # batch_size=1 inference, and information leaked between samples during
        # training.
        lo = magnitude.amin(dim=(1, 2, 3), keepdim=True)
        hi = magnitude.amax(dim=(1, 2, 3), keepdim=True)
        magnitude = (magnitude - lo) / (hi - lo + 1e-8)

        return magnitude