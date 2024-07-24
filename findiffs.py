#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: WinterSchool

File: findiffs.py
    
Description: some finite difference schemes. 
    
"""

import numpy as np
import matplotlib.pyplot as plt


__author__ = "Francois Lauze, University of Copenhagen"  
__date__ = "Thu Jan  3 11:22:42 2019"
__version = "0.0.0"


def right(m):
    return list(range(1,m)) + [m-1]

def left(m):
    return [0] + list(range(0,m-1))

def cdiff_x(phi):
    m = phi.shape[0]
    return 0.5*(phi[right(m)] - phi[left(m)])

def cdiff_y(phi):
    m = phi.shape[1]
    return 0.5*(phi[:, right(m)] - phi[:, left(m)])

def cdiff_z(phi):
    m = phi.shape[2]
    return 0.5*(phi[:, :, right(m)] - phi[:, :, left(m)])
    
def fdiff_x(phi):
    m = phi.shape[0]
    return phi[right(m)] - phi

def fdiff_y(phi):
    m = phi.shape[1]
    return phi[:, right(m)] - phi

def fdiff_z(phi):
    m = phi.shape[2]
    return phi[:, :, right(m)] - phi

def bdiff_x(phi):
    m = phi.shape[0]
    return phi - phi[left(m)]

def bdiff_y(phi):
    m = phi.shape[1]
    return phi - phi[:, left(m)]

def bdiff_z(phi):
    m = phi.shape[2]
    return phi - phi[:, :, left(m)]

def cdiff_xy(phi):
    m, n = phi.shape[:2]
    return 0.25*(phi[right(m)][:,right(n)] + phi[left(m)][:,left(n)] - 
                     phi[right(m)][:,left(n)] - phi[left(m)][:,right(n)])       

def cdiff_xz(phi):
    m, n = phi.shape[0], phi[2]
    return 0.25*(phi[right(m)][:,:,right(n)] + phi[left(m)][:,:,left(n)] - 
                     phi[right(m)][:,:,left(n)] - phi[left(m)][:,:,right(n)])       
def cdiff_yz(phi):
    m, n = phi.shape[1], phi[2]
    return 0.25*(phi[:,right(m)][:,:,right(n)] + phi[:,left(m)][:,:,left(n)] - 
                     phi[:,right(m)][:,:,left(n)] - phi[:,left(m)][:,:,right(n)])       

def cdiff_xx(phi):
    m = phi.shape[0]
    return phi[right(m)] -2*phi + phi[left(m)]

def cdiff_yy(phi):
    m = phi.shape[1]
    return phi[:,right(m)] -2*phi + phi[:,left(m)]

def cdiff_zz(phi):
    m = phi.shape[2]
    return phi[:,:,right(m)] -2*phi + phi[:,:,left(m)]




def finite_diffs2D(phi, i, j):
    """
    central, forward and backward differences for 2D phi
    """
    d1, d2 = phi.shape
    i_p = i - 1 if i > 0 else 0
    i_n = i + 1 if i < d1 - 1 else d1 - 1 
    j_p = j - 1 if j > 0 else 0
    j_n = j + 1 if j < d2 - 1 else d2 -1
    phi_cc, phi_nc, phi_pc, phi_cn, phi_cp = (
            phi[i, j], 
            phi[i_n, j], phi[i_p, j], 
            phi[i, j_n], phi[i, j_p])
    
    return (0.5*(phi_nc - phi_pc), 
            0.5*(phi_cn - phi_cp), 
            phi_nc - phi_cc, 
            phi_cc - phi_pc, 
            phi_cn - phi_cc, 
            phi_cc - phi_cp)


def finite_diffs3D(phi, i, j, k):
    """
    central, forward and backward differences for 3D phi
    """
    d1, d2, d3 = phi.shape
    i_p = i - 1 if i > 0 else 0
    i_n = i + 1 if i < d1 - 1 else d1 - 1 
    j_p = j - 1 if j > 0 else 0
    j_n = j + 1 if j < d2 - 1 else d2 -1
    k_p = k - 1 if k > 0 else 0
    k_n = k + 1 if k < d3 - 1 else d3 - 1
    phi_ccc, phi_ncc, phi_pcc, phi_cnc, phi_cpc, phi_ccn, phi_ccp = (
            phi[i, j, k], 
            phi[i_n, j, k], phi[i_p, j, k], 
            phi[i, j_n, k], phi[i, j_p, k], 
            phi[i, j, k_n], phi[i, j, k_p])
    return (0.5*(phi_ncc - phi_pcc), 
            0.5*(phi_cnc - phi_cpc), 
            0.5*(phi_ccn - phi_ccp), 
            phi_ncc - phi_ccc, 
            phi_ccc - phi_pcc, 
            phi_cnc - phi_ccc, 
            phi_ccc - phi_cpc, 
            phi_ccn - phi_ccc, 
            phi_ccc - phi_ccp)
    
