from abc import ABC
import numpy as np
import cv2


class BaseTransform(ABC):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class ToFloat(BaseTransform):
    def __init__(self, pytorch=True):
        super(ToFloat, self).__init__()
        self.pytorch = pytorch

    def __call__(self, x):
        if self.pytorch:
            return x.astype(np.float32, copy=False)
        else:
            return float(x)


class DivideByScalar(BaseTransform):
    def __init__(self, scalar, axis=None, channels=None):
        super(DivideByScalar, self).__init__()
        self.scalar = scalar
        self.axis = axis
        self.channels = channels

    def __call__(self, x):
        if self.axis is not None and self.channels:
            # axis is dimension to index into
            # channels is list of channels on that axis to divide
            idx = [slice(None)] * x.ndim
            for c in self.channels:
                idx[self.axis] = c
                x[tuple(idx)] /= self.scalar
        else:
            x /= self.scalar

        return x


class NormalizeToRange(BaseTransform):
    def __init__(self, prev_min, prev_max, new_min, new_max):
        super(NormalizeToRange, self).__init__()
        self.prev_min = prev_min
        self.prev_max = prev_max
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, x):
        # return (x - self.prev_min) / (self.prev_max - self.prev_min) * (self.new_max - self.new_min) + self.new_min
        # same as above but using inplace operations
        x -= self.prev_min
        x /= (self.prev_max - self.prev_min)
        x *= (self.new_max - self.new_min)
        x += self.new_min
        return x


class Reshape(BaseTransform):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def __call__(self, x):
        return x.reshape(self.shape)
    

class MoveAxis(BaseTransform):
    def __init__(self, orig_axis, new_axis):
        super(MoveAxis, self).__init__()
        self.orig_axis = orig_axis
        self.new_axis = new_axis

    def __call__(self, x):
        return np.moveaxis(x, self.orig_axis, self.new_axis)


class Rescale(BaseTransform):
    def __init__(self, scale):
        super(Rescale, self).__init__()
        self.scale = scale

    def __call__(self, x):
        return np.array([cv2.resize(frame, None, fx=self.scale, fy=self.scale) for frame in x])
    

class ClampToMax(BaseTransform):
    def __init__(self, max_val):
        super(ClampToMax, self).__init__()
        self.max_val = max_val

    def __call__(self, x):
        return np.clip(x, x.min(), self.max_val, out=x)