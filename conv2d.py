import numpy as np
from pathlib import Path
from PIL import Image
import math


class Conv2d:
    def __init__(
        self, img: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0
    ) -> None:
        self.img = self._get_img(img, padding)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.kernel_size = len(self.kernel)

    def convolve(self) -> np.ndarray:
        self._check_mismatch(self.img, len(self.kernel), self.stride)
        new_size = math.ceil((len(self.img) - (self.kernel_size - 1)) / self.stride)
        row_start = 0
        row_end = self.kernel_size

        if self.img.ndim == 3:
            conv = np.zeros((new_size, new_size, 3))
        else:
            conv = np.zeros((new_size, new_size))

        for i in range(new_size):
            col_start = 0
            col_end = self.kernel_size

            for j in range(new_size):
                if self.img.ndim == 3:
                    conv_entry = np.multiply(
                        self.img[row_start:row_end, col_start:col_end], self.kernel
                    )
                    new_px_value = self._get_new_px_value(conv_entry)
                    conv[i, j, :] = new_px_value
                else:
                    conv_entry = np.multiply(
                        self.img[row_start:row_end, col_start:col_end], self.kernel
                    ).sum()
                    conv[i, j] = conv_entry

                col_start += self.stride
                col_end += self.stride

            row_start += self.stride
            row_end += self.stride

        return conv

    def _get_img(self, img: np.ndarray, padding: int) -> np.ndarray:
        if img.ndim > 3:
            raise ValueError(f"Image can't have more than 3 dimensions, got {img.ndim}")
        elif img.ndim == 3 and img.shape[-1] > 3:
            raise ValueError(
                f"Image can't have more than 3 color channels, got {img.shape[-1]}"
            )
        elif padding < 0:
            raise ValueError("Padding can't have negative value")
        elif padding == 0:
            return img
        else:
            new_size = len(img) + padding * 2
            start = padding
            end = start + len(img)

            if img.ndim == 3:
                padded_img = np.zeros((new_size, new_size, 3))
                padded_img[start:end, start:end, :] = img
            elif img.ndim == 2:
                padded_img = np.zeros((new_size, new_size))
                padded_img[start:end, start:end] = img

            return padded_img

    def _get_new_px_value(self, conv_entry: np.ndarray) -> np.ndarray:
        px_value = np.empty(0)
        for i in range(3):
            px_value = np.append(px_value, np.sum(conv_entry[:, :, i]))

        return px_value

    def _check_mismatch(self, img: np.ndarray, kernel_size: int, stride: int) -> None:
        if kernel_size > len(img):
            raise ValueError("Kernel size can't be greater than X")
        elif kernel_size + stride > len(img):
            raise ValueError(
                f"Kernel size {kernel_size} and stride {stride} not compatible with input shape {img.shape}"
            )
        elif (
            stride % 2 == 0
            and kernel_size % 2 == 0
            and stride != 1
            and len(img) % 2 != 0
        ) or (
            stride % 2 != 0
            and kernel_size % 2 != 0
            and stride != 1
            and len(img) % 2 != 0
        ):
            raise ValueError(
                f"Stride length {stride} isn't compatible with kernel size {kernel_size} and input shape {img.shape}"
            )


