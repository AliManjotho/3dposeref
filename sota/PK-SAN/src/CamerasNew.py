
"""Utilities to deal with the cameras of human3.6m"""
from __future__ import division

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'res_w': 1000,
        'res_h': 1002
    },
    {
        'id': '55011271',
        'res_w': 1000,
        'res_h': 1000
    },
    {
        'id': '58860488',
        'res_w': 1000,
        'res_h': 1000
    },
    {
        'id': '60457274',
        'res_w': 1000,
        'res_h': 1002
    },
]



def project_point_radial( P, R, T, f, c, k, p ):
  """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3

  N = P.shape[0]
  X = R.dot( P.T - T ) # rotate and translate
  XX = X[:2,:] / X[2,:]
  r2 = XX[0,:]**2 + XX[1,:]**2

  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) )
  tan = 2*p[0]*XX[1,:] + 2*p[1]*XX[0,:]

  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

  Proj = (f * XXX) + c
  Proj = Proj.T

  D = X[2,]

  return Proj, D, radial, tan, r2

def world_to_camera_frame(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.dot( P.T - T ) # rotate and translate

  return X_cam.T

def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.T.dot( P.T ) + T # rotate and translate

  return X_cam.T

def load_camera_params( hf, path ):
  """Load h36m camera parameters

  Args
    hf: hdf5 open file with h36m cameras data
    path: path or key inside hf to the camera we are interested in
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
  """

  R = hf[ path.format('R') ][:]
  R = R.T

  T = hf[ path.format('T') ][:]
  f = hf[ path.format('f') ][:]
  c = hf[ path.format('c') ][:]
  k = hf[ path.format('k') ][:]
  p = hf[ path.format('p') ][:]

  name = hf[ path.format('Name') ][:]
  name = "".join( [chr(int(item)) for item1 in name for item in item1] )

  return R, T, f, c, k, p, name

def load_cameras( bpath , subjects=[1,5,6,7,8,9,11] ):
    """Loads the cameras of h36m

    Args
        bpath: path to hdf5 file with h36m camera data
        subjects: List of ints representing the subject IDs for which cameras are requested
    Returns
        rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
    """
    rcams = {}

    with h5py.File(bpath,'r') as hf:
        for s in subjects:
            for c in range(4): # There are 4 cameras in human3.6m
                R, T, f, cen, k, p, name = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )
                center = cen.reshape((1,2))
                focal_length = f.reshape((1,2))

                assert(name == h36m_cameras_intrinsic_params[c]['id'])

                center = normalize_screen_coordinates(center, 
                    h36m_cameras_intrinsic_params[c]['res_w'],
                    h36m_cameras_intrinsic_params[c]['res_h'])

                focal_length = focal_length / h36m_cameras_intrinsic_params[c]['res_w'] * 2.0
        
                tmp = np.concatenate((center,focal_length),axis=1)

                rcams[(s, c+1)] = (R, T, f, cen, k, p, name, tmp)

    return rcams

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]
