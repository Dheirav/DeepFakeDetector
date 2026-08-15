import os

from diagram_utils import NodeStyle, draw_edge, draw_node, new_figure, save_png


def main():
    output_dir = os.path.join("docs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = new_figure(figsize=(12.5, 3.8), dpi=300)
    base = NodeStyle()

    folders = draw_node(ax, "folders", "Dataset folders\n(train/val/test)", (0.10, 0.55), (0.20, 0.12), base)
    load = draw_node(ax, "load", "Image loading", (0.30, 0.55), (0.16, 0.12), base)
    bgr_rgb = draw_node(ax, "bgr_rgb", "BGR → RGB\nconversion", (0.48, 0.55), (0.18, 0.12), base)
    alb = draw_node(ax, "alb", "Albumentations\ntransforms", (0.66, 0.55), (0.18, 0.12), base)
    dataloader = draw_node(ax, "dataloader", "DataLoader\nbatching", (0.82, 0.55), (0.16, 0.12), base)
    batch = draw_node(ax, "batch", "Mini-batch\ntensors", (0.92, 0.20), (0.18, 0.12), base)

    draw_edge(ax, folders, load)
    draw_edge(ax, load, bgr_rgb)
    draw_edge(ax, bgr_rgb, alb)
    draw_edge(ax, alb, dataloader)
    draw_edge(ax, dataloader, batch, src_side="bottom", dst_side="top")

    save_png(fig, os.path.join(output_dir, "data_pipeline.png"), dpi=300)


if __name__ == "__main__":
    main()

