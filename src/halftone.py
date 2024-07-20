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
    colors=None,
    angles=None,
    alphas=None,
    lws=None,
    pad=0,
    use_black=False,
    fit_how="fit",
    dot_fill_how="circles"
):
    """
    Function to generate a halftone svg from a given image.
    The image is split into four layers (C, M, Y, K).
    Each layer consists of points on a grid.
    The size of the points is proportional to the average color intensity of the area covered by the circle.
    Grids are rotated to a given angle to mitigate the moiré effect.

    image_path: path to the image
    save_path: path to save the svg
    paper_w: width of the paper in meters
    paper_h: height of the paper in meters
    max_dot_size: maximum size of the dots in meters
    colors: list of hex color codes for each layer (default: ["#00ffff", "#ff00ff", "#ffff00", "#000000"])
    angles: list of angles in degrees for each layer (default: [15, 75, 0, 45])
    alphas: list of transparencies for each layer (default: [1, 1, 1, 1])
    lws: list of line widths in millimeters for each layer, adjust for pen thickness (default: [1, 1, 1, 1])
    pad: padding in meters (default: 0)
    use_black: whether to plot the black layer or not (default: False)
    fit_how: how to fit the image to the paper (default: "fit"), options: "fit", "fit-width", "fit-height"
    dot_fill_how: how to plot the dot, options: "lines", "circles" (default: "circles")
    """
    # Default values
    if colors is None:
        colors = ["#00ffff", "#ff00ff", "#ffff00", "#000000"]
    if angles is None:
        angles = [15, 75, 0, 45]
    if alphas is None:
        alphas = [1, 1, 1, 1]
    if lws is None:
        lws = [1, 1, 1, 1]
    alphas = np.array(alphas, dtype=float)
    tones = (1 / alphas) / (1 / alphas).max() / 2
    lws = np.array(lws) / 1000  # convert to meters
    # Load image
    image = load_image(image_path, use_black)
    image_h, image_w = fit_image(image, paper_h, paper_w, fit_how)
    convolved_layers = convolve(image, image_h, max_dot_size)
    # Plot
    mpl.rcParams["savefig.pad_inches"] = 0
    fig = plt.figure(figsize=(m_to_in(paper_w), m_to_in(paper_h)))
    fig.set_frameon(False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(-paper_w / 2, paper_w / 2)
    ax.set_ylim(-paper_h / 2, paper_h / 2)
    for i in range(4):
        points = generate_grid(
            angles[i], max_dot_size, paper_h, paper_w, image_h, image_w, pad
        )
        patches = list(
            chain.from_iterable(
                plot_dot(
                    p,
                    lws[i],
                    get_dot_radius(
                        p,
                        convolved_layers[i],
                        paper_w,
                        paper_h,
                        image_h,
                        image_w,
                        max_dot_size,
                        lws[i],
                        tones[i],
                    ),
                    dot_fill_how
                )
                for p in points
            )
        )
        coll = mpl.collections.PatchCollection(
            patches, edgecolor=colors[i], facecolor="none", linewidths=m_to_pt(lws[i]), alpha=alphas[i]
        )
        coll.set_gid("CMYK"[i])  # set the layer name in the svg
        coll.set_clip_on(False)
        ax.add_collection(coll)
        if not os.path.exists(f"{save_path}"):
            os.mkdir(f"{save_path}")
    fig.savefig(
        f'{save_path}/{image_path.split("/")[-1].split(".")[0]}_cmyk.svg',
        format="svg",
        pad_inches=0,
        transparent=True,
    )
