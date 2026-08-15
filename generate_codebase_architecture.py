import os

from diagram_utils import (
    EdgeStyle,
    NodeStyle,
    draw_edge,
    draw_group_label,
    draw_node,
    new_figure,
    save_png,
)


def main():
    output_dir = os.path.join("docs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = new_figure(figsize=(14, 6), dpi=300)

    base = NodeStyle()
    light = NodeStyle(facecolor="#f5f5f5")

    # Layout (left-to-right). Coordinates in [0,1]×[0,1]
    raw = draw_node(ax, "raw", "Raw datasets", (0.08, 0.70), (0.16, 0.10), base)
    dataset_builder = draw_node(ax, "dataset_builder", "dataset_builder/", (0.25, 0.70), (0.20, 0.10), base)

    train = draw_node(ax, "train", "train/", (0.42, 0.82), (0.14, 0.09), base)
    val = draw_node(ax, "val", "val/", (0.42, 0.70), (0.14, 0.09), base)
    test = draw_node(ax, "test", "test/", (0.42, 0.58), (0.14, 0.09), base)

    # scripts group (center)
    draw_group_label(ax, "scripts/", (0.56, 0.92), fontsize=11)
    scripts_root = draw_node(ax, "scripts_root", "scripts/", (0.56, 0.70), (0.16, 0.10), light)
    scripts_data = draw_node(ax, "scripts_data", "data/", (0.56, 0.52), (0.16, 0.09), base)
    scripts_pre = draw_node(ax, "scripts_pre", "preprocessing/", (0.56, 0.40), (0.16, 0.09), base)
    scripts_loader = draw_node(ax, "scripts_loader", "dataloader/", (0.56, 0.28), (0.16, 0.09), base)
    scripts_training = draw_node(ax, "scripts_training", "training/", (0.72, 0.70), (0.16, 0.10), base)
    scripts_eval = draw_node(ax, "scripts_eval", "evaluation/", (0.72, 0.52), (0.16, 0.09), base)
    scripts_exp = draw_node(ax, "scripts_exp", "explainability/", (0.72, 0.40), (0.16, 0.09), base)

    models = draw_node(ax, "models", "models/", (0.88, 0.70), (0.16, 0.10), base)
    frontend = draw_node(ax, "frontend", "frontend/", (0.88, 0.52), (0.16, 0.09), base)
    results = draw_node(ax, "results", "results/", (0.88, 0.34), (0.16, 0.09), base)
    logs = draw_node(ax, "logs", "logs/", (0.88, 0.18), (0.16, 0.08), base)

    # Data-flow arrows requested
    draw_edge(ax, raw, dataset_builder)
    draw_edge(ax, dataset_builder, train)
    draw_edge(ax, dataset_builder, val)
    draw_edge(ax, dataset_builder, test)

    draw_edge(ax, train, scripts_training, label="train", src_side="right", dst_side="left")
    draw_edge(ax, val, scripts_training, label="val", src_side="right", dst_side="left")

    draw_edge(ax, scripts_training, models, label="checkpoints")
    draw_edge(ax, models, scripts_eval, label="evaluate", src_side="bottom", dst_side="right")
    draw_edge(ax, models, frontend, label="serve", src_side="left", dst_side="right")
    draw_edge(ax, scripts_eval, results, label="metrics, plots", src_side="right", dst_side="left")

    # Internal scripts relations (light/dashed)
    dashed = EdgeStyle(linestyle="dashed")
    draw_edge(ax, scripts_root, scripts_data, style=dashed, src_side="bottom", dst_side="top")
    draw_edge(ax, scripts_root, scripts_pre, style=dashed, src_side="bottom", dst_side="top")
    draw_edge(ax, scripts_root, scripts_loader, style=dashed, src_side="bottom", dst_side="top")
    draw_edge(ax, scripts_root, scripts_training, style=dashed)
    draw_edge(ax, scripts_root, scripts_eval, style=dashed)
    draw_edge(ax, scripts_root, scripts_exp, style=dashed)

    # General logging edges (dashed)
    draw_edge(ax, dataset_builder, logs, style=dashed, src_side="bottom", dst_side="top")
    draw_edge(ax, scripts_training, logs, style=dashed, src_side="bottom", dst_side="top")
    draw_edge(ax, scripts_training, results, style=dashed, src_side="bottom", dst_side="top")

    save_png(fig, os.path.join(output_dir, "codebase_architecture.png"), dpi=300)


if __name__ == "__main__":
    main()

