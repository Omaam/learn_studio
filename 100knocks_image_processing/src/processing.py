"""Functions created by Omaam.
"""
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd


def convert_binarization(img: np.ndarray,
                         threshold: int) -> np.ndarray:
    img_in = img.copy()
    img_out = convert_rgb2gray(img_in)
    img_out = np.where(img_in < threshold, 0, 255)
    return img_out


def convert_rgb2gray(img):
    img_in = img.copy().astype(np.float32)
    rgb_weights = [0.2126, 0.7152, 0.0722]
    img_out = np.average(img_in, axis=2, weights=rgb_weights)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)
    return img_out
