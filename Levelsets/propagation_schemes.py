#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Rainbow Winter School 2019

File: propagation_schemes.py
    
Description: Implement elementary propagation schemes for levelsets.
Follow Sethian's Levelset Methods and Fast Marching Methods  
"""


import numpy as np
import findiffs as fd
MAX = np.maximum
MIN = np.minimum

__author__ = "Francois Lauze, University of Copenhagen"  
__date__ = "Sat Jan  5 00:19:18 2019"
__version = "0.0.0"


def FOSC2D_term(phi, F0):
    """First Order Space Convex 2D for propagation expansion velocity
        phi is the level set function
        F0 is the propagation expansion normal velocity
    """
    Dp = np.sqrt(MAX(fd.bdiff_x(phi), 0)**2 + MIN(fd.fdiff_x(phi), 0)**2 +
                 MAX(fd.bdiff_y(phi), 0)**2 + MIN(fd.fdiff_y(phi), 0)**2)
    
    Dm = np.sqrt(MAX(fd.fdiff_x(phi), 0)**2 + MIN(fd.bdiff_x(phi), 0)**2 +
                 MAX(fd.fdiff_y(phi), 0)**2 + MIN(fd.bdiff_y(phi), 0)**2)
    
    return (MAX(F0, 0)*Dp + MIN(F0, 0)*Dm)


def FOSC3D_term(phi, F0):
    """First Order Space Convex 3D for propagation expansion velocity
        phi is the level set function
        F0 is the propagation expansion normal velocity
    """
    Dp = np.sqrt(MAX(fd.bdiff_x(phi), 0)**2 + MIN(fd.fdiff_x(phi), 0)**2 +
                 MAX(fd.bdiff_y(phi), 0)**2 + MIN(fd.fdiff_y(phi), 0)**2 +
                 MAX(fd.bdiff_z(phi), 0)**2 + MIN(fd.fdiff_z(phi), 0)**2)
    
    Dm = np.sqrt(MAX(fd.fdiff_x(phi), 0)**2 + MIN(fd.bdiff_x(phi), 0)**2 +
                 MAX(fd.fdiff_y(phi), 0)**2 + MIN(fd.bdiff_y(phi), 0)**2 + 
                 MAX(fd.fdiff_z(phi), 0)**2 + MIN(fd.bdiff_z(phi), 0)**2)
    
    return (MAX(F0, 0)*Dp + MIN(F0, 0)*Dm)


def m(x, y):
    xy = x*y
    ax = np.abs(x)
    ay = np.abs(y)
    return np.logical_and(xy >= 0, ax <= ay)*x + np.logical_and(xy >= 0, ax > ay)*y
    
    

def SOSC2D_Term(phi, F0, dx, dy):
    """Second Order Space Convex 2D for propagation expansion velocity
        phi is the level set function
        F0 is the propagation expansion normal velocity
    """
    A = fd.bdiff_x(phi) + 0.5*dx*m(fd.bdiff_x(fd.bdiff_x(phi)), fd.cdiff_xx(phi))
    B = fd.bdiff_x(phi) - 0.5*dx*m(fd.fdiff_x(fd.fdiff_x(phi)), fd.cdiff_xx(phi))
    C = fd.bdiff_y(phi) + 0.5*dy*m(fd.bdiff_y(fd.bdiff_y(phi)), fd.cdiff_yy(phi))
    D = fd.bdiff_y(phi) - 0.5*dy*m(fd.fdiff_y(fd.fdiff_y(phi)), fd.cdiff_yy(phi))
    
    Dp = np.sqrt(MAX(A, 0)**2 + MIN(B, 0)**2 + 
                 MAX(C, 0)**2 + MIN(D, 0)**2)
    
    Dm = np.sqrt(MAX(B, 0)**2 + MIN(A, 0)**2 + 
                 MAX(D, 0)**2 + MIN(C, 0)**2)
    
    return (MAX(F0, 0)*Dp + MIN(F0, 0)*Dm)
    

def SOSC3D_Term(phi, F0, dx, dy, dz):
    """Second Order Space Convex 3D for propagation expansion velocity
        phi is the level set function
        F0 is the propagation expansion normal velocity
    """
    A = fd.bdiff_x(phi) + 0.5*dx*m(fd.bdiff_x(fd.bdiff_x(phi)), fd.cdiff_xx(phi))
    B = fd.bdiff_x(phi) - 0.5*dx*m(fd.fdiff_x(fd.fdiff_x(phi)), fd.cdiff_xx(phi))
    C = fd.bdiff_y(phi) + 0.5*dy*m(fd.bdiff_y(fd.bdiff_y(phi)), fd.cdiff_yy(phi))
    D = fd.bdiff_y(phi) - 0.5*dy*m(fd.fdiff_y(fd.fdiff_y(phi)), fd.cdiff_yy(phi))
    E = fd.bdiff_z(phi) + 0.5*dz*m(fd.bdiff_z(fd.bdiff_z(phi)), fd.cdiff_zz(phi))
    F = fd.bdiff_z(phi) - 0.5*dz*m(fd.fdiff_z(fd.fdiff_z(phi)), fd.cdiff_zz(phi))
    
    Dp = np.sqrt(MAX(A, 0)**2 + MIN(B, 0)**2 + 
                 MAX(C, 0)**2 + MIN(D, 0)**2 + 
                 MAX(E, 0)**2 + MIN(F, 0)**2)
    
    Dm = np.sqrt(MAX(B, 0)**2 + MIN(A, 0)**2 + 
                 MAX(D, 0)**2 + MIN(C, 0)**2 + 
                 MAX(F, 0)**2 + MIN(E, 0)**2)
    
    return (MAX(F0, 0)*Dp + MIN(F0, 0)*Dm)
    

def upwind2D_Term(phi, Ux, Uy):
    """2D upwind scheme for advection velocity
    """
    return (MAX(Ux, 0)*fd.bdiff_x(phi) + MIN(Ux, 0)*fd.fdiff_x(phi) +
            MAX(Uy, 0)*fd.bdiff_y(phi) + MIN(Uy, 0)*fd.fdiff_y(phi))
    

def upwind3D_Term(phi, Ux, Uy, Uz):
    """3D upwind scheme for advection velocity
    """
    return (MAX(Ux, 0)*fd.bdiff_x(phi) + MIN(Ux, 0)*fd.fdiff_x(phi) +
            MAX(Uy, 0)*fd.bdiff_y(phi) + MIN(Uy, 0)*fd.fdiff_y(phi) + 
            MAX(Uz, 0)*fd.bdiff_z(phi) + MIN(Uy, 0)*fd.fdiff_z(phi))
    


def curvature2D_Term(phi, G, eps=1e-5):
    """curvature dependent velocity term 2D."""
    cx = fd.cdiff_x(phi)
    cy = fd.cdiff_y(phi)
    cxx = fd.cdiff_xx(phi)
    cyy = fd.cdiff_yy(phi)
    cxy = fd.cdiff_xy(phi)
    return G*(cxx*cy**2 + cyy*cx**2 - 2*cx*cy*cxy)/(cx**2 + cy**2 + eps)
    

def curvature3D_Term(phi, G, eps=1e-5):
    """Mean curvature dependent velocity term in 3D."""
    cx = fd.cdiff_x(phi)
    cy = fd.cdiff_y(phi)
    cz = fd.cdiff_z(phi)
    cxx = fd.cdiff_xx(phi)
    cyy = fd.cdiff_yy(phi)
    czz = fd.cdiff_zz(phi)
    cxy = fd.cdiff_xy(phi)
    cxz = fd.cdiff_xz(phi)
    cyz = fd.cdiff_yz(phi)
    
    return G*((cxx + cyy)*cz**2 + (cxx + czz)*cy*2 + (cyy + czz)*cx*2 - 
            2*(cx*cy*cxy + cx*cz*cxz + cy*cz*cyz))/(cx**2 + cy**2 + cz**2 + eps)
     
def curvature2D(phi, eps=1e-5):
    """curvature from levelset function."""
    cx = fd.cdiff_x(phi)
    cy = fd.cdiff_y(phi)
    cxx = fd.cdiff_xx(phi)
    cyy = fd.cdiff_yy(phi)
    cxy = fd.cdiff_xy(phi)
    return (cxx*cy**2 + cyy*cx**2 - 2*cx*cy*cxy)/np.power(cx**2 + cy**2 + eps, 1.5)
    

def curvature3D(phi, eps=1e-5):
    """Mean curvature from levelset function."""
    cx = fd.cdiff_x(phi)
    cy = fd.cdiff_y(phi)
    cz = fd.cdiff_z(phi)
    cxx = fd.cdiff_xx(phi)
    cyy = fd.cdiff_yy(phi)
    czz = fd.cdiff_zz(phi)
    cxy = fd.cdiff_xy(phi)
    cxz = fd.cdiff_xz(phi)
    cyz = fd.cdiff_yz(phi)
    
    return ((cxx + cyy)*cz**2 + (cxx + czz)*cy*2 + (cyy + czz)*cx*2 - 
            2*(cx*cy*cxy + cx*cz*cxz + cy*cz*cyz))/np.power(cx**2 + cy**2 + cz**2 + eps, 1.5)
     
    
    
    