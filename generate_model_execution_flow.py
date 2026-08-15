import os

from diagram_utils import EdgeStyle, NodeStyle, draw_edge, draw_node, new_figure, save_png


def main():
    output_dir = os.path.join("docs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = new_figure(figsize=(14, 4.5), dpi=300)
    base = NodeStyle()
    dashed = EdgeStyle(linestyle="dashed")

    input_img = draw_node(ax, "input", "Input image", (0.08, 0.60), (0.15, 0.11), base)
    rgb = draw_node(ax, "rgb", "RGB\npreprocessing", (0.22, 0.60), (0.16, 0.11), base)

    srm = draw_node(ax, "srm", "SRM layer", (0.36, 0.75), (0.14, 0.10), base)
    fft = draw_node(ax, "fft", "FFT layer", (0.36, 0.45), (0.14, 0.10), base)

    concat = draw_node(ax, "concat", "Channel\nconcatenation", (0.50, 0.60), (0.18, 0.11), base)
    backbone = draw_node(ax, "backbone", "Backbone\nnetwork", (0.66, 0.60), (0.16, 0.11), base)
    attention = draw_node(ax, "attention", "Attention\nmodule", (0.78, 0.60), (0.15, 0.11), base)
    pool = draw_node(ax, "pool", "Pooling", (0.88, 0.60), (0.13, 0.11), base)
    fc = draw_node(ax, "fc", "Fully connected\nclassifier", (0.88, 0.35), (0.18, 0.11), base)
    softmax = draw_node(ax, "softmax", "Softmax\nprobabilities", (0.88, 0.15), (0.18, 0.11), base)

    draw_edge(ax, input_img, rgb)
    draw_edge(ax, rgb, srm, src_side="right", dst_side="left")
    draw_edge(ax, rgb, fft, src_side="right", dst_side="left")
    draw_edge(ax, srm, concat, src_side="right", dst_side="left")
    draw_edge(ax, fft, concat, src_side="right", dst_side="left")
    draw_edge(ax, rgb, concat, style=dashed, label="(optional)")

    draw_edge(ax, concat, backbone)
    draw_edge(ax, backbone, attention)
    draw_edge(ax, attention, pool)
    draw_edge(ax, pool, fc, src_side="bottom", dst_side="top")
    draw_edge(ax, fc, softmax, src_side="bottom", dst_side="top")

    save_png(fig, os.path.join(output_dir, "model_execution_flow.png"), dpi=300)


if __name__ == "__main__":
    main()

