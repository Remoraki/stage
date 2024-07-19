#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Rainbow Winter School 2019

File: redistance.py
    
Description: implement a classical redistance function in pure Python.
Python rewriting of older Matlab implementation adapted from a CImg 
implementation. To be compared with scikitfmm.distance?

Pure Python code, not as performant as C++, Cython etc, but portable.


"""
    
__author__ = "Francois Lauze, University of Copenhagen"  
__date__ = "Thu Jan  3 11:22:42 2019"
__version = "0.1.0"


import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import hypot
from findiffs import cdiff_x, cdiff_y, cdiff_z
from findiffs import fdiff_x, fdiff_y, fdiff_z
from findiffs import bdiff_x, bdiff_y, bdiff_z
from findiffs import finite_diffs2D, finite_diffs3D
 

def redistance2D(phi, iterations, time_step, band):
    """A classical redistance function, solving 
    dphi/dt = -sign(phi)(|nabla\phi|-1)
    Written in pure Python Numpy for portability 
    Parameters:
    -----------
    phi: ndarray
        2D array: levelset function.
    iterations: int
        number of redistancing iterations
    time_step: float
        algorithmic time step
    band: float
        narrow band size as distance to 0 level set. A
        0 or negative value means the entire image.
    Returns:
    --------
    redistanced phi
    """
    if band <= 0:
        dcphi = np.zeros(phi.shape + (2,))
        fphi  = np.zeros(phi.shape + (2,))
        bphi  = np.zeros(phi.shape + (2,))
        uwphi = np.zeros(phi.shape + (2,))
        for i in range(iterations):
            dcphi[:,:,0] = cdiff_x(phi)
            dcphi[:,:,1] = cdiff_y(phi)
            
            fphi[:,:,0] = fdiff_x(phi)
            fphi[:,:,1] = fdiff_y(phi)
            
            bphi[:,:,0] = bdiff_x(phi)
            bphi[:,:,1] = bdiff_y(phi)
            
            sgn = -np.sign(phi)
            sgn.shape = sgn.shape + (1,) # remember: array broadcasting rules
            # upwind scheme
            uwphi = np.maximum(0, dcphi*sgn)*fphi - np.minimum(0, dcphi*sgn)*bphi
            ngrad = np.linalg.norm(dcphi, axis=-1) + 1e-5
            ngrad.shape = ngrad.shape + (1,) # broadcasting again!
            dcphi /= ngrad
            sgn.shape = sgn.shape[:-1] # and again!
            velocity = sgn*(uwphi[:,:,0]*dcphi[:,:,0] + uwphi[:,:,1]*dcphi[:,:,1] - 1.0) 
            vmax = np.abs(velocity).max()
            if vmax > 0:
                phi += (time_step/vmax)*velocity
    else: # does it pay off in Python?
        velocity = np.zeros_like(phi)
        vmax = 0
        m, n = phi.shape
        for i in range(iterations):
            for x,y in product(range(m), range(n)):
                phic = phi[x, y]
                if abs(phic) < band:
                    gx, gy, fx, bx, fy, by = finite_diffs2D(phi, x, y)
                    sgn = -1 if phic > 0 else 1 if phic < 0 else 0
                    ix = fx if sgn*gx > 0 else bx
                    iy = fy if sgn*gy > 0 else by
                    ng = hypot(gx, gy) + 1e-5
                    ngx, ngy = gx/ng, gy/ng
                    veloc = sgn*(ix*ngx + iy*ngy - 1)
                    velocity[x, y] = veloc
                    aveloc = abs(veloc)
                    if aveloc > vmax:
                        vmax  = aveloc
            if vmax > 0:
                phi += (time_step/vmax)*velocity
    return phi

                 
                
        
def redistance3D(phi, iterations, time_step, band):
    """A classical redistance function, solving 
    dphi/dt = -sign(phi)(|nabla\phi|-1)
    Written in pure Python Numpy for portability 
    Parameters:
    -----------
    phi: ndarray
        2D array: levelset function.
    iterations: int
        number of redistancing iterations
    time_step: float
        algorithmic time step
    band: float
        narrow band size as distance to 0 level set. A
        0 or negative value means the entire image.
    Returns:
    --------
    redistanced phi
    """
    if band <= 0:
        dcphi = np.zeros(phi.shape + (3,))
        fphi  = np.zeros(phi.shape + (3,))
        bphi  = np.zeros(phi.shape + (3,))
        uwphi = np.zeros(phi.shape + (3,))
        for i in range(iterations):
            dcphi[:,:,:,0] = cdiff_x(phi)
            dcphi[:,:,:,1] = cdiff_y(phi)
            dcphi[:,:,:,2] = cdiff_z(phi)
            
            fphi[:,:,:,0] = fdiff_x(phi)
            fphi[:,:,:,1] = fdiff_y(phi)
            fphi[:,:,:,2] = fdiff_z(phi)
            
            bphi[:,:,:,0] = bdiff_x(phi)
            bphi[:,:,:,1] = bdiff_y(phi)
            bphi[:,:,:,2] = bdiff_z(phi)
            
            sgn = -np.sign(phi)
            sgn.shape = sgn.shape + (1,) # remember: array broadcasting rules
            uwphi = np.maximum(0, dcphi*sgn)*fphi - np.minimum(0, dcphi*sgn)*bphi

            ngrad = np.linalg.norm(dcphi, axis=-1) + 1e-5
            ngrad.shape = ngrad.shape + (1,) # remember: array broadcasting rules 
            dcphi /= ngrad
            sgn.shape = sgn.shape[:-1]
            velocity = sgn*(uwphi[:,:,:,0]*dcphi[:,:,:,0] + uwphi[:,:,:,1]*dcphi[:,:,:,1] + uwphi[:,:,:,2]*dcphi[:,:,:,2] - 1)
            vmax = np.abs(velocity).max()
            if vmax > 0:
                phi += (time_step/vmax)*velocity
    else:
        velocity = np.zeros_like(phi)
        vmax = 0
        m, n, p = phi.shape
        for i in range(iterations):
            for x,y,z in product(range(m), range(n), range(p)):
                phic = phi[x, y, z]
                if abs(phic) < band:
                    gx, gy, gz, fx, bx, fy, by, fz, bz = finite_diffs3D(phi, x, y, z)
                    sgn = -1 if phic > 0 else 1 if phic < 0 else 0
                    ix = fx if sgn*gx > 0 else bx
                    iy = fy if sgn*gy > 0 else by
                    iz = fz if sgn*gz > 0 else bz
                    ng = np.linalg.norm((gx, gy, gz)) + 1e-5
                    ngx, ngy, ngz = gx/ng, gy/ng, gz/ng
                    veloc = sgn*(ix*ngx + iy*ngy + iz*ngz- 1)
                    velocity[x, y, z] = veloc
                    aveloc = abs(veloc)
                    if aveloc > vmax:
                        vmax  = aveloc
            if vmax > 0:
                phi += (time_step/vmax)*velocity
    return phi



def redistance(phi, iterations, time_step=0.5, band=0.0):    
    """A classical redistance function, solving 
    dphi/dt = -sign(phi)(|nabla\phi|-1)
    Written in pure Python Numpy for portability 
    Parameters:
    -----------
    phi: ndarray
        2D or 3D array: levelset function.
    iterations: int
        number of redistancing iterations
    time_step: float
        algorithmic time step
    band: float
        narrow band size as distance to 0 level set. A
        0 or negative value means the entire image.
    Returns:
    --------
    redistanced phi
    """
    f = {2: redistance2D, 3:redistance3D}
    if phi.ndim not in [2,3]:
        raise ValueError('Levelset function must be 2D or 3D.')
    return f[phi.ndim](phi.copy(), iterations, time_step=time_step, band=band)


    
        
            
if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    x, y = np.mgrid[-1:1:101j, -1:1:101j]
    c = ((x**2 + y**2) < 0.25).astype(float) - 0.5
    phi = redistance(-c, 50)
    ax0 = plt.subplot(121, projection='3d')
    ax1 = plt.subplot(122, projection='3d')  
    ax0.plot_surface(x, y, c)
    ax1.plot_surface(x, y, phi)  
    plt.show()    
    














           

