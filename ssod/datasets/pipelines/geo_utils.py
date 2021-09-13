"""
Record the geometric transformation information used in the augmentation in a transformation matrix.
"""
import numpy as np


class GeometricTransformationBase(object):
    @classmethod
    def inverse(cls, results):
        # compute the inverse
        return results["transform_matrix"].I  # 3x3

    @classmethod
    def apply(self, results, operator, **kwargs):
        trans_matrix = getattr(self, f"_get_{operator}_matrix")(**kwargs)
        if "transform_matrix" not in results:
            results["transform_matrix"] = trans_matrix
        else:
            base_transformation = results["transform_matrix"]
            results["transform_matrix"] = np.dot(trans_matrix, base_transformation)

    @classmethod
    def apply_cv2_matrix(self, results, cv2_matrix):
        if cv2_matrix.shape[0] == 2:
            mat = np.concatenate(
                [cv2_matrix, np.array([0, 0, 1]).reshape((1, 3))], axis=0
            )
        else:
            mat = cv2_matrix
        base_transformation = results["transform_matrix"]
        results["transform_matrix"] = np.dot(mat, base_transformation)
        return results

    @classmethod
    def _get_rotate_matrix(cls, degree=None, cv2_rotation_matrix=None, inverse=False):
        # TODO: this is rotated by zero point
        if degree is None and cv2_rotation_matrix is None:
            raise ValueError(
                "At least one of degree or rotation matrix should be provided"
            )
        if degree:
            if inverse:
                degree = -degree
            rad = degree * np.pi / 180
            sin_a = np.sin(rad)
            cos_a = np.cos(rad)
            return np.array([[cos_a, sin_a, 0], [-sin_a, cos_a, 0], [0, 0, 1]])  # 2x3
        else:
            mat = np.concatenate(
                [cv2_rotation_matrix, np.array([0, 0, 1]).reshape((1, 3))], axis=0
            )
            if inverse:
                mat = mat * np.array([[1, -1, -1], [-1, 1, -1], [1, 1, 1]])
            return mat

    @classmethod
    def _get_shift_matrix(cls, dx=0, dy=0, inverse=False):
        if inverse:
            dx = -dx
            dy = -dy
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    @classmethod
    def _get_shear_matrix(
        cls, degree=None, magnitude=None, direction="horizontal", inverse=False
    ):
        if magnitude is None:
            assert degree is not None
            rad = degree * np.pi / 180
            magnitude = np.tan(rad)

        if inverse:
            magnitude = -magnitude
        if direction == "horizontal":
            shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0], [0, 0, 1]])
        else:
            shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0], [0, 0, 1]])
        return shear_matrix

    @classmethod
    def _get_flip_matrix(cls, shape, direction="horizontal", inverse=False):
        h, w = shape
        if direction == "horizontal":
            flip_matrix = np.float32([[-1, 0, w], [0, 1, 0], [0, 0, 1]])
        else:
            flip_matrix = np.float32([[1, 0, 0], [0, h - 1, 0], [0, 0, 1]])
        return flip_matrix

    @classmethod
    def _get_scale_matrix(cls, sx, sy, inverse=False):
        if inverse:
            sx = 1 / sx
            sy = 1 / sy
        return np.float32([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
