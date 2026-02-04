from typing import TypeVar, Generic

import numpy as np

DType = TypeVar("DType")


class Array(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:
        return super().__getitem__(key)  # type: ignore
