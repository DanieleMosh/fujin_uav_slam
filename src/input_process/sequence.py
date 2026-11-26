from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class Frame:
    """One camera frame. Images are already undistorted, so K alone describes the camera."""
    timestamp: float                    # seconds
    image_bgr: np.ndarray               # HxWx3 uint8
    K: np.ndarray                       # 3x3 pinhole intrinsics
    T_wc: Optional[np.ndarray] = None   # 4x4 camera-to-world pose, if known
    depth: Optional[np.ndarray] = None  # HxW float32 metres, 0 = invalid, if known


class Sequence(ABC):
    """A dataset that yields frames in time order."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        pass
