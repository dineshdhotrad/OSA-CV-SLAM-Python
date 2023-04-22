import numpy as np
from pyquaternion import Quaternion


class simpleQuaternion(Quaternion):
    def rotate(self, p):
        """ Function to rotate points p by q

        Args:
            p (np.ndarray): [N, 3] - 3D points.

        Returns:
            np.ndarray: [N, 3] - rotated points.
        """
        q = self.elements
        rq, vq = q[0], q[1:]  # 1, [3, ]
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
        return f'simpleQuaternion{self.w, self.x, self.y, self.z}'

    def __repr__(self):
        return f'simpleQuaternion{self.w, self.x, self.y, self.z}'

def getQuaternion(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    cross = cross / np.linalg.norm(cross)

    nv1 = v1 / np.linalg.norm(v1)
    nv2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(nv1, nv2)
    angle1 = np.arccos(dot_product) #angle in radian

    return simpleQuaternion(axis=cross, angle=angle1)

def multiplyQuadernion(q1, q2):
    return q2 * q1

def get_quaternion_from_euler(roll, pitch, yaw):
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

  return simpleQuaternion([qw, qx, qy, qz])

def axis_transformation(points):
    # axis transformation 
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    points = np.array([x,y,z]).T
    return points