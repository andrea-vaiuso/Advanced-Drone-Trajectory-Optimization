import numpy as np

def wrap_angle(angle: float) -> float:
    """
    Wrap an angle in the range [-pi, pi].

    Parameters:
        angle (float): Angle in radians.

    Returns:
        float: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def euler_to_rot(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert Euler angles (roll=phi, pitch=theta, yaw=psi) into a rotation matrix.
    Assumes the rotation order Rz(psi) * Ry(theta) * Rx(phi).

    Parameters:
        phi (float): Roll angle in radians.
        theta (float): Pitch angle in radians.
        psi (float): Yaw angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi),  np.cos(psi), 0],
                   [0, 0, 1]])
    Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi),  np.cos(phi)]])
    return Rz @ Ry @ Rx