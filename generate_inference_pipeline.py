import os

from diagram_utils import EdgeStyle, NodeStyle, draw_edge, draw_node, new_figure, save_png


def main():
    output_dir = os.path.join("docs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = new_figure(figsize=(14, 4.5), dpi=300)
    base = NodeStyle()

    upload = draw_node(ax, "upload", "User upload", (0.08, 0.62), (0.15, 0.11), base)
    validate = draw_node(ax, "validate", "Image validation", (0.22, 0.62), (0.17, 0.11), base)
    preprocess = draw_node(ax, "preprocess", "Preprocessing", (0.38, 0.62), (0.16, 0.11), base)
    infer = draw_node(ax, "infer", "Model inference", (0.54, 0.62), (0.16, 0.11), base)
    softmax = draw_node(ax, "softmax", "Softmax\nprediction", (0.70, 0.62), (0.16, 0.11), base)
    ui = draw_node(ax, "ui", "User interface\ndisplay", (0.88, 0.62), (0.18, 0.11), base)

    gradcam = draw_node(ax, "gradcam", "GradCAM\ngeneration", (0.62, 0.25), (0.18, 0.11), base)
    overlay = draw_node(ax, "overlay", "Heatmap\noverlay", (0.80, 0.25), (0.16, 0.11), base)

    draw_edge(ax, upload, validate)
    draw_edge(ax, validate, preprocess)
    draw_edge(ax, preprocess, infer)
    draw_edge(ax, infer, softmax)
    draw_edge(ax, softmax, ui, label="class +\nprobabilities")

    dashed = EdgeStyle(linestyle="dashed")
    draw_edge(ax, infer, gradcam, label="features +\nlogits", src_side="bottom", dst_side="top", style=dashed)
    draw_edge(ax, gradcam, overlay)
    draw_edge(ax, overlay, ui, label="explanations", src_side="top", dst_side="bottom")

    save_png(fig, os.path.join(output_dir, "inference_pipeline.png"), dpi=300)


if __name__ == "__main__":
    main()

