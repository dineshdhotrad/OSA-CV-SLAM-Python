import numpy as np
from src.software.utils.featureMods import modQuaternion
from pyquaternion import Quaternion


def get3DProjection(color_image, depth_image, camera_intrinsics):
    """ creates 3D point cloud of rgb images by taking depth information

        Args : color image (numpy array[h, w, c], dtype= uint8): 
                depth image (numpy array[h, w]) : values of all channels will be same

        Returns:
            np.ndarray: [3, n]: Exports 3D points converted from Depth Image
            np.ndarray: [3, n]: Exports RGB data for corresponding 3D points
        """

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                  np.linspace(0,image_height-1,image_height))
    camera_points_x = np.multiply(pixel_x-camera_intrinsics[0,2],depth_image/camera_intrinsics[0,0])
    camera_points_y = np.multiply(pixel_y-camera_intrinsics[1,2],depth_image/camera_intrinsics[1,1])
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]

    return camera_points,color_points

def getMod3DPoint(point, quad_rotation, translation):
    """Get the 3d points modified according to provided rotation and translation

    Args:
        point (numpy array[3, 1]): 3D point
        quad_rotation (numpy array[4, 1]): Quadranion Rotaton Value [W, X, Y, Z]
        translation (numpy array[3, 1]): Translation Value

    Returns:
        np.ndarray: [3, n]: Exports 3D points converted WRT R,t
    """

    rotation = Quaternion(quad_rotation)
    modPoint = rotation.rotate(point)
    modPoint = modPoint + translation
    return modPoint

def getMod3DPoints(points, quad_rotation, translation):
    """Get the 3d points modified according to provided rotation and translation

    Args:
        points (numpy array[3, n]): Array of 3D points
        quad_rotation (numpy array[4, 1]): Quadranion Rotaton Value [W, X, Y, Z]
        translation (numpy array[3, 1]): Translation Value

    Returns:
        np.ndarray: [3, n]: Exports 3D points converted WRT R,t
    """

    rotation = modQuaternion(quad_rotation)
    modPoints = rotation.rotate(points)
    modPoints = modPoints + translation
    return modPoints

def getIsometry(rotation, translation):
    """

    Args:
        rotation (_type_): _description_
        translation (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.vstack([np.hstack([rotation,translation.reshape(3,-1)]), [0, 0, 0, 1]])