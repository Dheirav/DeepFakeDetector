#!/usr/bin/env python3
"""
Generate chapter figures for CHAPTER 4: METHODOLOGY.

Produces a set of PNGs under the output directory showing:
 - overall pipeline diagram (network graph)
 - SRM kernels as heatmaps (from scripts/preprocessing/srm.py)
 - GeM pooling behaviour for several p values
 - Example FFT magnitude (from a dataset image if available, otherwise synthetic)

Run from repository root:
    python scripts/figures/generate_chapter_figures.py --out docs/figures/
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception:
    torch = None


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def gen_pipeline_diagram(outdir: Path, dpi: int = 300, vector: bool = False):
    try:
        import networkx as nx
    except Exception:
        nx = None

    path_png = outdir / "pipeline.png"
    path_svg = outdir / "pipeline.svg"
    if nx is None:
        plt.figure(figsize=(12, 3), dpi=dpi)
        plt.text(0.5, 0.5, "Pipeline diagram requires networkx; install with pip install networkx",
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(path_png, bbox_inches='tight', dpi=dpi)
        if vector:
            plt.savefig(path_svg, bbox_inches='tight')
        plt.close()
        return

    G = nx.DiGraph()
    nodes = [
        "Raw Sources",
        "Index / Validate",
        "Deduplicate / Filter",
        "Sample / Balance",
        "Cluster Split / Export",
        "Train / Val / Test",
        "Training (PreprocessNet -> Backbone)",
        "Evaluation / Model Cards"
    ]
    for n in nodes:
        G.add_node(n)
    edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
    G.add_edges_from(edges)

    # Prefer Graphviz layout for clarity when available
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 4), dpi=dpi)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='#88ccee', arrowsize=20, font_size=10)
    plt.title('Overall System Pipeline')
    plt.savefig(path_png, bbox_inches='tight', dpi=dpi)
    if vector:
        plt.savefig(path_svg, bbox_inches='tight', format='svg')
    plt.close()


def gen_srm_kernels(outdir: Path, dpi: int = 300, vector: bool = False):
    path_png = outdir / "srm_kernels.png"
    path_svg = outdir / "srm_kernels.svg"
    if torch is None:
        plt.figure(figsize=(8, 3), dpi=dpi)
        plt.text(0.5, 0.5, "Torch not available; cannot render SRM kernels.", ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(path_png, bbox_inches='tight', dpi=dpi)
        if vector:
            plt.savefig(path_svg, bbox_inches='tight')
        plt.close()
        return

    try:
        from scripts.preprocessing.srm import SRMLayer
    except Exception as e:
        plt.figure(figsize=(8, 3), dpi=dpi)
        plt.text(0.5, 0.5, f"Failed to import SRMLayer: {e}", ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(path_png, bbox_inches='tight', dpi=dpi)
        if vector:
            plt.savefig(path_svg, bbox_inches='tight')
        plt.close()
        return

    srm = SRMLayer()
    kernels = srm.kernels.detach().cpu().numpy()  # [3,1,5,5]
    kernels = kernels[:, 0, :, :]

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=dpi)
    vabs = max(abs(kernels).max(), 1e-6)
    for i in range(3):
        im = axs[i].imshow(kernels[i], cmap='seismic', interpolation='nearest', vmin=-vabs, vmax=vabs)
        axs[i].set_title(f'SRM kernel {i+1}')
        axs[i].axis('off')
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    plt.suptitle('SRM High-pass Kernels (Laplacian, Horizontal, Vertical)')
    plt.savefig(path_png, bbox_inches='tight', dpi=dpi)
    if vector:
        plt.savefig(path_svg, bbox_inches='tight', format='svg')
    plt.close()


def gem_pooling_behavior(outdir: Path, dpi: int = 300, vector: bool = False):
    path_png = outdir / "gem_behavior.png"
    path_svg = outdir / "gem_behavior.svg"
    # Use the repository's GeM implementation and a realistically-structured
    # activation tensor (sparse local peaks) so the plot matches behaviour from
    # the codebase.
    try:
        from scripts.modules.attention_heads import GeM
        if torch is None:
            raise RuntimeError('torch not available')
    except Exception as e:
        # fallback to simple illustrative plot
        rng = np.random.RandomState(0)
        patches = rng.rand(100, 16)
        p_vals = [1.0, 2.0, 3.0, 8.0]
        pooled = {}
        for p in p_vals:
            pooled[p] = np.mean(patches ** p, axis=1) ** (1.0 / p)

        plt.figure(figsize=(8, 4), dpi=dpi)
        for p in p_vals:
            plt.plot(sorted(pooled[p]), label=f'p={p}')
        plt.title('GeM pooling: pooled values distribution for different p (fallback)')
        plt.xlabel('sorted patch index')
        plt.ylabel('pooled value')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.savefig(path_png, bbox_inches='tight', dpi=dpi)
        if vector:
            plt.savefig(path_svg, bbox_inches='tight', format='svg')
        plt.close()
        return

    # Build a synthetic but realistic activation map: mostly low values with a
    # handful of strong local activations per channel (to mimic feature maps).
    B, C, H, W = 1, 16, 8, 8
    act = torch.rand(B, C, H, W) * 0.1
    rng = np.random.RandomState(1)
    # add sparse high activations
    for ch in range(C):
        for _ in range(3):
            y = int(rng.randint(0, H))
            x = int(rng.randint(0, W))
            act[0, ch, y, x] += float(rng.rand() * 0.9 + 0.1)

    p_vals = [1.0, 2.0, 3.0, 8.0]
    results = {}
    for p in p_vals:
        gem = GeM(p=float(p), learnable=False)
        gem.eval()
        with torch.no_grad():
            out = gem(act).squeeze().cpu().numpy()  # shape: [C]
        results[p] = np.sort(out)

    plt.figure(figsize=(8, 4), dpi=dpi)
    for p in p_vals:
        plt.plot(results[p], label=f'p={p}')
    plt.title('GeM pooling: pooled per-channel values for different p (repo GeM)')
    plt.xlabel('sorted channel index')
    plt.ylabel('pooled value')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(path_png, bbox_inches='tight', dpi=dpi)
    if vector:
        plt.savefig(path_svg, bbox_inches='tight', format='svg')
    plt.close()


def example_fft(outdir: Path, data_root: Path, dpi: int = 300, vector: bool = False):
    path = outdir / "fft_example.png"
    path_svg = outdir / "fft_example.svg"
    # try to use a real image from dataset
    img_path = None
    for cls in ("real", "ai_generated", "ai_edited"):
        d = data_root / cls
        if d.exists() and any(d.iterdir()):
            img_path = next(d.iterdir())
            break

    if img_path is None:
        # make synthetic image
        im = np.zeros((256, 256), dtype=np.float32)
        rr, cc = np.ogrid[:256, :256]
        im += (rr - 128) ** 2 + (cc - 128) ** 2
        im = (im - im.min()) / (im.max() - im.min())
    else:
        try:
            from PIL import Image
            im = Image.open(img_path).convert('L').resize((256, 256))
            im = np.array(im).astype(np.float32) / 255.0
        except Exception:
            im = np.random.rand(256, 256)

    # Prefer using the repository's FFTLayer so the visualisation matches code.
    mag = None
    if torch is not None:
        try:
            from scripts.preprocessing.fft import FFTLayer
            # prepare tensor: B x 3 x H x W
            t = torch.from_numpy(im).float().unsqueeze(0).unsqueeze(0)  # 1x1xHxW
            t = t.repeat(1, 3, 1, 1)
            with torch.no_grad():
                fft_layer = FFTLayer()
                mag_t = fft_layer(t)
            mag = mag_t.squeeze().cpu().numpy()
        except Exception:
            mag = None

    if mag is None:
        F = np.fft.fftshift(np.fft.fft2(im))
        mag = np.log(np.abs(F) + 1e-8)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    plt.figure(figsize=(8, 4), dpi=dpi)
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.title('Example image (grayscale)')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mag, cmap='inferno')
    plt.title('FFT magnitude (log)')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
    if vector:
        plt.savefig(path_svg, bbox_inches='tight', format='svg')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate chapter figures')
    parser.add_argument('--out', type=str, default='docs/figures/', help='Output directory')
    parser.add_argument('--data-root', type=str, default='dataset_builder/train', help='Dataset root (optional)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PNG outputs')
    parser.add_argument('--vector', action='store_true', help='Also save SVG vector outputs')
    args = parser.parse_args()

    outdir = Path(args.out)
    ensure_outdir(outdir)

    gen_pipeline_diagram(outdir, dpi=args.dpi, vector=args.vector)
    gen_srm_kernels(outdir, dpi=args.dpi, vector=args.vector)
    gem_pooling_behavior(outdir, dpi=args.dpi, vector=args.vector)
    example_fft(outdir, Path(args.data_root), dpi=args.dpi, vector=args.vector)

    print(f"Figures written to: {outdir}")


if __name__ == '__main__':
    main()
