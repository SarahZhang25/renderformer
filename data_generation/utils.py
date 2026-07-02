"""
Copy of utils.py from Neural-Radiosity-Renderer
"""

import numpy as np
from typing import Dict, Tuple, Optional
import os


def tone_map_reinhard(image: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """
    Apply Reinhard tone mapping to HDR image.

    Args:
        image: HDR image array (any shape, values can exceed 1)
        exposure: Exposure multiplier

    Returns:
        Tone-mapped image in [0, 1] range
    """
    scaled = image * exposure
    return scaled / (1.0 + scaled)


def tone_map_gamma(image: np.ndarray, gamma: float = 2.2, exposure: float = 1.0) -> np.ndarray:
    """
    Apply simple gamma correction tone mapping.

    Args:
        image: HDR image array
        gamma: Gamma value (typically 2.2 for sRGB)
        exposure: Exposure multiplier

    Returns:
        Tone-mapped image in [0, 1] range
    """
    scaled = np.clip(image * exposure, 0, None)
    return np.power(scaled, 1.0 / gamma)


def tone_map(image: np.ndarray, method: str = "reinhard", **kwargs) -> np.ndarray:
    """
    Apply tone mapping with specified method.

    Args:
        image: HDR image array
        method: "reinhard" or "gamma"
        **kwargs: Additional arguments for the specific method

    Returns:
        Tone-mapped image in [0, 1] range
    """
    if method == "reinhard":
        return tone_map_reinhard(image, **kwargs)
    elif method == "gamma":
        return tone_map_gamma(image, **kwargs)
    else:
        raise ValueError(f"Unknown tone mapping method: {method}")


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert float image [0, 1] to uint8 [0, 255].

    Args:
        image: Float image in [0, 1] range

    Returns:
        uint8 image
    """
    # TODO: remove these debug messages after fixing this...
    nan_count = np.isnan(image).sum()
    posinf_count = np.isposinf(image).sum()
    neginf_count = np.isneginf(image).sum()
    
    if nan_count > 0 or posinf_count > 0 or neginf_count > 0:
        print(f"[to_uint8] Cleaning image: {nan_count} NaNs, {posinf_count} +Infs, {neginf_count} -Infs")
        

    clean_image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    clipped = np.clip(clean_image, 0, 1)
    return (clipped * 255).astype(np.uint8)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1/2.4) - a)


def save_image(image: np.ndarray, path: str, tone_mapping: str = "reinhard", exposure: float = 1.0, save=True):
    """
    Save image to file with tone mapping.

    Args:
        image: HDR image array (height, width, 3)
        path: Output file path
        tone_mapping: Tone mapping method ("reinhard", "gamma", or "none")
        exposure: Exposure value for tone mapping
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Apply tone mapping
    if tone_mapping == "none":
        processed = np.clip(image * exposure, 0, 1)  # or just image if already [0,1]
    else:
        processed = tone_map(image, method=tone_mapping, exposure=exposure)

    # Clamp to [0, 1]
    processed = np.clip(processed, 0, 1)
    
    processed = linear_to_srgb(processed)

    # # Apply sRGB gamma correction (linear -> display)
    # processed = np.power(image, 1.0 / 2.2)

    # Convert to uint8
    img_uint8 = to_uint8(processed)

    # Save using available library
    if save:
        try:
            from PIL import Image
            pil_image = Image.fromarray(img_uint8)
            pil_image.save(path)
        except ImportError:
            try:
                import imageio
                imageio.imwrite(path, img_uint8)
            except ImportError:
                import matplotlib.pyplot as plt
                plt.imsave(path, img_uint8)
    else:
        return img_uint8

