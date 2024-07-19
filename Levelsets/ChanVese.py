#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Winter School 2019

File: ChanVese.py
    
Description: Demo Chan-Vese with Level sets
    
"""

import numpy as np
import matplotlib.pyplot as plt
from redistance import redistance
from propagation_schemes import curvature2D
from typing import Tuple, Union

__author__ = "Francois Lauze, University of Copenhagen"  
__date__ = "Sun Jan  6 15:51:43 2019"
__version = "0.0.0"

PI = np.pi
frame_count = 1


def save_frame(fig: plt.Figure):
    global frame_count
    frame_name = 'figure_%03d.png' % frame_count
    fig.savefig(frame_name)
    frame_count += 1


def H(x : Union[float, np.ndarray], epsilon: float = 0.01) -> Union[float, np.ndarray]:
    """Regularized Heaviside.

    :param x: float or np.ndarray, input function
    :param epsilon: float, regularization parameter
    :return: float or np.ndarray, regularized Heaviside function
    """
    return 0.5*(1 + (2/PI)*np.arctan(x/epsilon))


def Dirac(x : Union[float, np.ndarray], epsilon: float = 0.01) -> Union[float, np.ndarray]:
    """Regularized Dirac, derivative of H.

    :param x: float or np.ndarray, input function
    :param epsilon: float, regularization parameter
    """
    return (2/PI)*epsilon/(epsilon**2 + x**2)
    

def means(u: np.ndarray, phi: np.ndarray, eps: float) -> Tuple[float, float]:
    """ Mean values for u at H(phi) and 1-H(phi).

    :param u: np.ndarray, image
    :param phi: np.ndarray, level set function
    :param eps: float, regularization parameter

    :return: Tuple[float, float], mean values of u at H(phi) and 1-H(phi)
    """
    co = ((u*H(phi, eps)).sum())/(H(phi, eps).sum())
    ci = ((u*H(-phi, eps)).sum())/(H(-phi, eps).sum())
    return co, ci


def CVIterations(u: np.ndarray, phi: np.ndarray,
                 mu: float, nu: float, outer_iterations: int = 200,
                 inner_iterations: int = 10,
                 redistancing: int = 3, dt: float = 0.5, eps: float = 0.1,
                 ax: Union[None, plt.Axes] = None, cnt=None, save_the_frames: bool = False) -> Tuple[np.ndarray, float, float]:
    """
    Chan-Vese iterations.
    :param u: np.ndarray, image
    :param phi: np.ndarray, level set function
    :param mu: float, regularization parameter
    :param nu: float, data weight
    :param outer_iterations: int, number of outer iterations,
            where the level set function is redistanced.
    :param inner_iterations: int, number of inner iterations,
            where the level set function is updated and the means are computed.
    :param redistancing: int, number of redistancing iterations.
    :param dt: float, time step
    :param eps: float, regularization parameter
    :param ax: Union[None, plt.Axes], axis to plot
    :param cnt: None, contour plot object.
    :param save_the_frames: bool, save the frames if True

    :return: Tuple[np.ndarray, float, float], level set function, mean value at H(phi), mean value at 1-H(phi)
    """

    if ax is None:
        fig, ax = plt.subplots(1,1)
        ax.imshow(u, cmap='gray')
        ax.axis('off')
        cnt = ax.contour(phi, [0], colors=['g'], linewidths=3.5)
        plt.pause(3)
        fig.canvas.draw()
    else:
        fig = ax.get_figure()

    for j in range(outer_iterations):
        phi = redistance(phi, iterations=redistancing)
        co, ci = means(u, phi, eps)
        for i in range(inner_iterations):
            dphi = Dirac(phi, eps)
            L1 = mu*curvature2D(phi, eps)
            L2 = nu*((u-ci)**2 - (u-co)**2)
            velocity = dphi*(L1 + L2)
            mveloc = np.abs(velocity).max()
            if abs(mveloc) > 1e-10:
                phi = phi + dt*velocity#/mveloc
                c0, ci = means(u, phi, eps)
            else:
                break
        
        for c in cnt.collections:
            c.remove()
        cnt = ax.contour(phi, [0], colors=['g'], linewidths=3.5)
        plt.pause(0.001)
        fig.canvas.draw()
        if save_the_frames:
            save_frame(fig)
    return phi, co, ci


def normalize_img(u: np.ndarray, new_min: float = 0.0, new_max: float = 1.0):
    """
    Normalize image.

    :param u: np.ndarray, image
    :param new_min: float, new minimum value
    :param new_max: float, new maximum value
    :return: np.ndarray, normalized image
    """
    old_min = u.min()
    old_max = u.max()
    a, b = np.dot(np.linalg.inv([[old_min, 1], [old_max, 1]]), (new_min, new_max))
    return a*u + b

    
def run_an_example(save_the_frames=False):
    """
    Run an example of Chan-Vese with Level sets.
    """
    v = np.load('scene.npy')
    phi0 = np.load('phi0.npy')
    v = v + np.random.randn(*v.shape)*20
    phi = phi0.copy()
    phi, ax, cnt = CVIterations(v, phi, mu=3.0e6, nu = 1.0e5, outer_iterations=300, dt=1.0e-9, save_the_frames=save_the_frames)
    plt.show()


if __name__ == '__main__':
    run_an_example(save_the_frames=False)

        
    