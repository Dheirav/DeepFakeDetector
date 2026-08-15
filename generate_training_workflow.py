import os

from diagram_utils import EdgeStyle, NodeStyle, draw_edge, draw_node, new_figure, save_png


def main():
    output_dir = os.path.join("docs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = new_figure(figsize=(14, 4.8), dpi=300)
    base = NodeStyle()
    dashed = EdgeStyle(linestyle="dashed")

    # Forward flow (left to right)
    dataset = draw_node(ax, "dataset", "Dataset\n(train / val)", (0.08, 0.60), (0.16, 0.11), base)
    loader = draw_node(ax, "loader", "DeepfakeDataset\nloader", (0.22, 0.60), (0.18, 0.11), base)
    alb = draw_node(ax, "alb", "Albumentations\naugmentations", (0.38, 0.60), (0.19, 0.11), base)
    srm_fft = draw_node(ax, "srm_fft", "SRM / FFT\npreprocessing", (0.54, 0.60), (0.18, 0.11), base)
    backbone = draw_node(ax, "backbone", "Backbone\nnetwork", (0.68, 0.60), (0.16, 0.11), base)
    attention = draw_node(ax, "attention", "Attention\nmodule", (0.80, 0.60), (0.15, 0.11), base)
    loss = draw_node(ax, "loss", "Loss\ncomputation", (0.92, 0.60), (0.15, 0.11), base)

    # Optimization and scheduling row (below)
    optimizer = draw_node(ax, "optimizer", "Optimizer", (0.78, 0.25), (0.14, 0.10), base)
    scheduler = draw_node(ax, "scheduler", "LR\nscheduler", (0.92, 0.25), (0.14, 0.10), base)
    checkpoint = draw_node(ax, "checkpoint", "Checkpoint\nsaving", (0.62, 0.25), (0.16, 0.10), base)

    # Forward edges
    draw_edge(ax, dataset, loader, label="samples")
    draw_edge(ax, loader, alb, label="images")
    draw_edge(ax, alb, srm_fft, label="augmented")
    draw_edge(ax, srm_fft, backbone, label="batch")
    draw_edge(ax, backbone, attention, label="features")
    draw_edge(ax, attention, loss, label="logits")

    # Backward/optimization flow
    draw_edge(ax, loss, optimizer, label="backprop\ngradients", src_side="bottom", dst_side="top")
    draw_edge(ax, optimizer, backbone, label="update\nweights", src_side="top", dst_side="bottom")
    draw_edge(ax, optimizer, scheduler, label="step", src_side="right", dst_side="left")
    draw_edge(ax, scheduler, optimizer, label="adjust LR", src_side="left", dst_side="right", style=dashed)

    # Checkpoint decision from validation metrics (conceptual dashed)
    draw_edge(ax, loss, checkpoint, label="best val\nmetrics", src_side="bottom", dst_side="top", style=dashed)

    save_png(fig, os.path.join(output_dir, "training_workflow.png"), dpi=300)


if __name__ == "__main__":
    main()

