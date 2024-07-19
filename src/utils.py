import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_paper_sizes(n=5):
    """
    n: maximum size of paper (default: 5)

    returns: lists of heights and widths of A-series paper sizes (in meters), index in list corresponds to size (A0, A1, A2, ...)
    """
    a = math.sqrt(2)
    b = 1
    hs = [1 / 2 ** (1 / 4)]
    ws = [2 ** (1 / 4)]
    for i in range(1, n):
        hs.append(ws[i - 1] / 2)
        ws.append(hs[i - 1])
    return hs, ws


def load_image(path, use_black=False, gray_tolerance=25):
    """
    path: path to the image
    use_black: use black layer from the image (default: False)
    gray_tolerance: tolerance for gray color (default: 25)

    returns: numpy array of shape (h, w, 4)
    """
    pil_img = Image.open(path)
    image = np.array(pil_img.convert("CMYK"))
    if use_black:
        mask = np.abs(image[:, :, 0] - image[:, :, 1]) <= gray_tolerance
        mask &= np.abs(image[:, :, 1] - image[:, :, 2]) <= gray_tolerance
        mask &= np.abs(image[:, :, 2] - image[:, :, 0]) <= gray_tolerance
        new_layer = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
        new_layer[mask] = image[mask].mean(axis=1)
        image[:, :, 3] = new_layer
        for i in range(3):
            image[mask, i] = 0
    return image


def fit_image(image, paper_h, paper_w, how="fit"):
    """
    image: numpy array of shape (h, w, 4)
    paper_h: height of the paper in meters
    paper_w: width of the paper in meters
    how: how to fit the image (default: "fit"), options: "fit", "fit-width", "fit-height"
    """
    image_h_px, image_w_px, _ = image.shape
    if how == "fit":
        scale = max(image_w_px / paper_w, image_h_px / paper_h)
    elif how == "fit-width":
        scale = image_w_px / paper_w
    elif how == "fit-height":
        scale = image_h_px / paper_h
    else:
        raise ValueError("Invalid value for 'how'")
    image_h = image_h_px / scale
    image_w = image_w_px / scale
    return image_h, image_w


def convolve(image, image_h, max_dot_size):
    """
    Calculates the average color intensity of the area covered by a halftone dot.

    image: numpy array of shape (h, w, 4)
    image_h: height of the image in meters
    max_dot_size: maximum size of the dots in meters

    returns: list of 4 numpy arrays of shape (h, w)
    """
    image_h_px, image_w_px, _ = image.shape
    layers = []
    k = int((max_dot_size / image_h) * image_h_px)
    for i in range(4):
        layers.append(
            convolve2d(image[:, :, i], np.ones((k, k)) / (k * k), mode="same")
        )
    return layers


def get_coord(point, paper_h, paper_w, image_h, image_w):
    """
    Shifts the origin of the coordinate system to the center of the paper.
    """
    y = point[0] + paper_w / 2 - (paper_w - image_w) / 2
    x = point[1] + paper_h / 2 - (paper_h - image_h) / 2
    return x, y


def is_valid(point, paper_h, paper_w, image_h, image_w, pad):
    """
    Checks if a point is within the bounds of the paper and padding.
    """
    x, y = get_coord(point, paper_h, paper_w, image_h, image_w)
    if x < 0 or x > image_h or y < 0 or y > image_w:
        return False
    x += (paper_h - image_h) / 2
    y += (paper_w - image_w) / 2
    if x < pad or y < pad or x > paper_h - pad or y > paper_w - pad:
        return False
    return True


def get_dot_radius(
    point, convolved_image, paper_w, paper_h, image_h, image_w, max_dot_size, lw, tone
):
    """
    Calculates the radius of a halftone dot based on the average color intensity of the area covered by the circle.

    point: coordinates of the point
    convolved_image: numpy array of shape (h, w), value of each pixel should be the average color intensity of the area covered by the circle
    paper_w: width of the paper in meters
    paper_h: height of the paper in meters
    image_h: height of the image in meters
    image_w: width of the image in meters
    max_dot_size: maximum size of the dots in meters
    lw: line width in meters
    tone: tone of the layer, used to adjust the intensity of the color

    returns: radius of the dot in meters
    """
    image_h_px, image_w_px = convolved_image.shape
    x, y = get_coord(point, paper_h, paper_w, image_h, image_w)
    x = image_h_px - int(x / image_h * image_h_px) - 1
    y = int(y / image_w * image_w_px)
    r = math.sqrt(
        convolved_image[x, y] / 255 * max_dot_size**2
    )  # Area scales quadratically with radius
    if r * tone < lw / 2.83465 / 1000 / 2:  # min radius is half the line width
        return 0
    return r * tone


def plot_dot(point, lw, radius):
    """
    Plots a halftone dot. The dot is drawn as a circle with a series of vertical lines to fill the circle.
    This is done because the plotter software does not support filled circles.

    point: coordinates of the point
    lw: line width in millimeters
    radius: radius of the dot in meters

    returns: list of patches to be added to the plot
    """
    lw_pt = lw / 2.83465 / 1000  # mm to pt
    x, y = point
    if radius == 0:
        return []
    l = radius - lw_pt / 2
    patches = [
        plt.Line2D(
            [x - l + a * lw_pt, x - l + a * lw_pt],
            [
                y - math.sqrt(l**2 - (l - a * lw_pt) ** 2),
                y + math.sqrt(l**2 - (l - a * lw_pt) ** 2),
            ],
        )
        for a in range(0, int(math.ceil(2 * l / lw_pt)))
        if math.sqrt(l**2 - (l - a * lw_pt) ** 2) > lw_pt / 2
    ]
    patches.append(plt.Circle(point, radius=radius))
    return patches


def generate_grid(angle, max_dot_size, paper_h, paper_w, image_h, image_w, pad=0):
    """
    Generates a list of point coordinates on a grid rotated to a given angle.
    """
    size = math.sqrt(paper_h**2 + paper_w**2)
    gamma = angle / 180 * math.pi  # to radians
    points = []
    for a in np.linspace(0, size, int(size // max_dot_size) + 1):
        for b in np.linspace(0, size, int(size // max_dot_size) + 1):
            p = (
                (-size / 2 + a) * math.cos(-gamma) + (-size / 2 + b) * math.sin(-gamma),
                (-size / 2 + b) * math.cos(-gamma) - (-size / 2 + a) * math.sin(-gamma),
            )
            if is_valid(p, paper_h, paper_w, image_h, image_w, pad):
                points.append(p)
    return points


def plot_dots(
    ax,
    convolved_image,
    points,
    paper_w,
    paper_h,
    image_h,
    image_w,
    max_dot_size,
    color,
    alpha,
    tone,
    lw,
):
    """
    Plots halftone dots on a grid.

    ax: matplotlib axis
    convolved_image: numpy array of shape (h, w), value of each pixel should be the average color intensity of the area covered by the circle
    points: list of point coordinates
    paper_w: width of the paper in meters
    paper_h: height of the paper in meters
    image_h: height of the image in meters
    image_w: width of the image in meters
    max_dot_size: maximum size of the dots in meters
    color: color of the dots (hex code)
    alpha: transparency of the dots
    tone: tone of the layer, used to adjust the intensity of the color
    lw: line width in meters
    """
    patches = list(
        chain.from_iterable(
            plot_dot(
                p,
                lw,
                get_dot_radius(
                    p,
                    convolved_image,
                    paper_w,
                    paper_h,
                    image_h,
                    image_w,
                    max_dot_size,
                    lw,
                    tone,
                ),
            )
            for p in points
        )
    )
    coll = mpl.collections.PatchCollection(
        patches, edgecolor=color, facecolor="none", linewidths=lw, alpha=alpha
    )
    ax.add_collection(coll)
