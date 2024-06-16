from typing import Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
from mmcv.transforms import TRANSFORMS
from mmcv.transforms.utils import cache_random_params
from mmcv.transforms.wrappers import *

# Define type of transform or transform config
Transform = Union[Dict, Callable[[Dict], Dict]]

# Indicator of keys marked by KeyMapper._map_input, which means ignoring the
# marked keys in KeyMapper._apply_transform so they will be invisible to
# wrapped transforms.
# This can be 2 possible case:
# 1. The key is required but missing in results
# 2. The key is manually set as ... (Ellipsis) in ``mapping``, which means
# the original value in results should be ignored
IgnoreKey = object()

# Import nullcontext if python>=3.7, otherwise use a simple alternative
# implementation.
try:
    from contextlib import nullcontext  # type: ignore
except ImportError:
    from contextlib import contextmanager

    @contextmanager  # type: ignore
    def nullcontext(resource=None):
        try:
            yield resource
        finally:
            pass


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img


@TRANSFORMS.register_module()
class MasaTransformBroadcaster(KeyMapper):
    """A transform wrapper to apply the wrapped transforms to multiple data
    items. For example, apply Resize to multiple images.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be wrapped.
        mapping (dict): A dict that defines the input key mapping.
            Note that to apply the transforms to multiple data items, the
            outer keys of the target items should be remapped as a list with
            the standard inner key (The key required by the wrapped transform).
            See the following example and the document of
            ``mmcv.transforms.wrappers.KeyMapper`` for details.
        remapping (dict): A dict that defines the output key mapping.
            The keys and values have the same meanings and rules as in the
            ``mapping``. Default: None.
        auto_remap (bool, optional): If True, an inverse of the mapping will
            be used as the remapping. If auto_remap is not given, it will be
            automatically set True if 'remapping' is not given, and vice
            versa. Default: None.
        allow_nonexist_keys (bool): If False, the outer keys in the mapping
            must exist in the input data, or an exception will be raised.
            Default: False.
        share_random_params (bool): If True, the random transform
            (e.g., RandomFlip) will be conducted in a deterministic way and
            have the same behavior on all data items. For example, to randomly
            flip either both input image and ground-truth image, or none.
            Default: False.

    """

    def __init__(
        self,
        transforms: List[Union[Dict, Callable[[Dict], Dict]]],
        mapping: Optional[Dict] = None,
        remapping: Optional[Dict] = None,
        auto_remap: Optional[bool] = None,
        allow_nonexist_keys: bool = False,
        share_random_params: bool = False,
    ):
        super().__init__(
            transforms, mapping, remapping, auto_remap, allow_nonexist_keys
        )

        self.share_random_params = share_random_params

    def scatter_sequence(self, data: Dict) -> List[Dict]:
        """Scatter the broadcasting targets to a list of inputs of the wrapped
        transforms."""

        # infer split number from input
        seq_len = 0
        key_rep = None

        if self.mapping:
            keys = self.mapping.keys()
        else:
            keys = data.keys()

        for key in keys:
            assert isinstance(data[key], Sequence)
            if seq_len:
                if len(data[key]) != seq_len:
                    raise ValueError(
                        "Got inconsistent sequence length: "
                        f"{seq_len} ({key_rep}) vs. "
                        f"{len(data[key])} ({key})"
                    )
            else:
                seq_len = len(data[key])
                key_rep = key

        assert seq_len > 0, "Fail to get the number of broadcasting targets"

        scatters = []
        for i in range(seq_len):  # type: ignore
            scatter = data.copy()
            for key in keys:
                scatter[key] = data[key][i]
            scatters.append(scatter)
        return scatters

    def transform(self, results: Dict):
        """Broadcast wrapped transforms to multiple targets."""

        # Apply input remapping
        inputs = self._map_input(results, self.mapping)

        # Scatter sequential inputs into a list
        input_scatters = self.scatter_sequence(inputs)

        # Control random parameter sharing with a context manager
        if self.share_random_params:
            # The context manager :func`:cache_random_params` will let
            # cacheable method of the transforms cache their outputs. Thus
            # the random parameters will only generated once and shared
            # by all data items.
            ctx = cache_random_params  # type: ignore
        else:
            ctx = nullcontext  # type: ignore

        with ctx(self.transforms):
            output_scatters = [
                self._apply_transforms(_input) for _input in input_scatters
            ]

        outputs = {
            key: [_output[key] for _output in output_scatters]
            for key in output_scatters[0]
        }

        # Apply remapping
        outputs = self._map_output(outputs, self.remapping)

        results.update(outputs)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(transforms = {self.transforms}"
        repr_str += f", mapping = {self.mapping}"
        repr_str += f", remapping = {self.remapping}"
        repr_str += f", auto_remap = {self.auto_remap}"
        repr_str += f", allow_nonexist_keys = {self.allow_nonexist_keys}"
        repr_str += f", share_random_params = {self.share_random_params})"
        return repr_str
