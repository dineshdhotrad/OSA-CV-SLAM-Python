import numpy as np
from pyquaternion import Quaternion

class modQuaternion(Quaternion):
    def rotate(self, p):
        """ Function to rotate points p by q

        Args:
            p (np.ndarray): [N, 3] - 3D points.

        Returns:
            np.ndarray: [N, 3] - rotated points.
        """
        quads = self.elements
        rq, vq = quads[0], quads[1:]  # 1, [3, ]
        vq_ = -vq  # [3, ]

        # p_ = qpq_ = (qp)q_, vw = -v.w + vxw
        # qp
        rqp = -np.dot(p, vq)  # [N, ]
        vqp = (rq*p) + np.cross(vq, p)  # [N, 3]

        # (qp)q_
        # rqpq = rqp*rq - np.dot(vqp, vq_)  # [N, ]
        vqpq = rqp[:, None]*vq_[None, :] + rq*vqp + np.cross(vqp, vq_)  # [N, 3]
        return vqpq

    def __str__(self):
        return f'modQuaternion{self.w, self.x, self.y, self.z}'

    def __repr__(self):
        return f'modQuaternion{self.w, self.x, self.y, self.z}'