#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: WinterSchool

File: SimpleGAC
    
Description:
    
"""

import numpy as np
import matplotlib.pyplot as plt

__author__ = "Francois Lauze, University of Copenhagen"  
__date__ = "Fri Jan  4 10:51:20 2019"
__version = "0.0.0"

from redistance import redistance
import propagation_schemes as ps
import findiffs as fd
from scalespace import gaussian_scalespace


def makeG(v, sigma, l):
    vx, vy = gaussian_scalespace(v, sigma=sigma, order=((1,0),(0,1)))
    ng = vx**2 + vy**2
    return np.exp(-ng/l**2)
    



def GACIterations(phi, G, Gx, Gy, m=300, n=10, redist=2, balloon=0.001, ax=None, cnt=None):
    """
    Run some GAC iterations.
    :G, Gx, Gy: are the metric/edge function and its drivatives
    :m: the number of 'outer' iterations: at each outer iteration,
      phi is redistanced.
    :n:  the bumber of inner iterations / levelset evolutions per 
        outer iteration
    :redist: iterations for the redistancing function
    :balloon: metric modulated balloon force
    :ax: axes where to display the evolution
    :cnt: contour collection for updating
    """
    if ax == None:
        _, ax = plt.subplots(1,1)
        ax.imshow(v, cmap='gray')
        cnt = ax.contour(phi, [0], colors=['r'], linewidths=2.5)
    for j in range(m):
        phi = redistance(phi, iterations=redist)
        L1 = np.zeros_like(phi)
        L2 = np.zeros_like(phi)
        L3 = np.zeros_like(phi)
        for i in range(n):
            # Advection by gradient of G
            L1 = ps.upwind2D_Term(phi, Gx, Gy)
            # G-weighted curvature flow
            L2 = ps.curvature2D_Term(phi, G)
            # Balloon force a.k.a propagation expansion 
            L3 = ps.FOSC2D_term(phi, G*balloon)
            veloc = L1 + L2 + L3
            mveloc = np.abs(veloc).max()
            print('mveloc=', mveloc)
            phi = phi + (1.0/mveloc)*(veloc)
        
        if (j % 5 == 0):
            for c in cnt.collections:
                c.remove()
            cnt = ax.contour(phi, [0], colors=['r'], linewidths=3.0)
            plt.pause(0.001)
            ax.figure.canvas.draw()
    return phi, ax, cnt

    
if __name__ == "__main__":
    v = np.load('scene.npy')
    phi0 = np.load('phi0.npy')
    #G = makeG(v, 3.0, 70)
    G = np.load('G.npy')
    Gx = fd.cdiff_x(G)
    Gy = fd.cdiff_y(G)
    plt.imshow(G)
    phi = phi0.copy()
    phi, ax, cnt = GACIterations(phi, G, Gx, Gy, m=1500, balloon=0.01)
    