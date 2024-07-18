import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.utils import *


def generate_halftone(
    image_path,
    save_path,
    paper_w,
    paper_h,
    max_dot_size,
    split=False,
    colors=None,
    angles=None,
    alphas=None,
    lws=None,
    pad=0,
):
    # Default values
    if colors is None:
        colors = ["#00ffff", "#ff00ff", "#ffff00", "#000000"]
    if angles is None:
        angles = [15, 75, 0, 45]
    if alphas is None:
        alphas = np.array([1, 1, 1, 1], dtype=float)
    if lws is None:
        lws = np.array([1, 1, 1, 1], dtype=float)
    # Conversions
    cm = 100 / 2.54
    tones = (1 / alphas) / (1 / alphas).max() / 2
    lws *= (1 / 25.4) * 72  # mm to pt (1mm = 1/25.4 inch, 1 inch = 72 pt)
    # Load image
    image = load_image(image_path)
    image_h, image_w = fit_image(image, paper_h, paper_w)
    convolved_image = convolve(image, image_h, max_dot_size)
    # Plot
    mpl.rcParams["savefig.pad_inches"] = 0
    fig = plt.figure(
        figsize=(paper_w * cm, paper_h * cm), dpi=100, layout="tight", facecolor="none"
    )
    plt.autoscale(tight=True)
    plt.axis("off")
    ax = fig.add_subplot(facecolor="none")
    ax.clear()
    for i in range(4):
        if split:
            ax.clear()
        ax.set_axis_off()
        ax.set_xlim(-paper_w / 2, paper_w / 2)
        ax.set_ylim(-paper_h / 2, paper_h / 2)
        points = generate_grid(
            angles[i], max_dot_size, paper_h, paper_w, image_h, image_w, pad
        )
        plot_dots(
            ax,
            convolved_image[i],
            points,
            paper_w,
            paper_h,
            image_h,
            image_w,
            max_dot_size,
            colors[i],
            alphas[i],
            tones[i],
            lws[i],
        )
        if not os.path.exists(f"{save_path}"):
            os.mkdir(f"{save_path}")
        if split:
            fig.savefig(
                f'{save_path}/{image_path.split(".")[0]}_{"cmyk"[i]}.svg',
                format="svg",
                pad_inches=0,
                bbox_inches="tight",
                transparent=True,
            )
    if not split:
        fig.savefig(
            f'{save_path}/{image_path.split(".")[0]}_cmyk.svg',
            format="svg",
            pad_inches=0,
            bbox_inches="tight",
            transparent=True,
        )
