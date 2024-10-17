# Ed's Code
import sys
sys.path.append("..")

import numpy as np
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
from astropy.io import fits
from pprint import pprint
from tqdm import tqdm
import pickle
import corner
import os

from scipy import optimize
from scipy.stats import poisson

import jax
import jax.numpy as jnp

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc_file('../utils/matplotlibrc')

from utils import ed_fcts_amarel as ef
from utils import create_mask as cm

def preprocess_map_shapes(rad = True):
    # required code to draw edges and fill regions in plot
    t = np.linspace(0, 2 * np.pi, 100)
    l_list = 20. * np.cos(t)
    b_list = 20. * np.sin(t)

    inner_roi_x = 20. * np.cos(t)
    inner_roi_y = 20. * np.sin(t)

    outer_roi_low_x = 30. * np.cos(t)
    outer_roi_low_y = 30. * np.sin(t)

    outer_roi_high_x = 40. * np.cos(t)
    outer_roi_high_y = 40. * np.sin(t)

    outer_roi_lim_x = 80. * np.cos(t)
    outer_roi_lim_y = 80. * np.sin(t)
    
    if rad == True:
        inner_roi_x = np.deg2rad(inner_roi_x)
        inner_roi_y = np.deg2rad(inner_roi_y)
        outer_roi_low_x = np.deg2rad(outer_roi_low_x)
        outer_roi_low_y = np.deg2rad(outer_roi_low_y)
        outer_roi_high_x = np.deg2rad(outer_roi_high_x)
        outer_roi_high_y = np.deg2rad(outer_roi_high_y)
        outer_roi_lim_x = np.deg2rad(outer_roi_lim_x)
        outer_roi_lim_y = np.deg2rad(outer_roi_lim_y)

    return l_list, b_list, inner_roi_x, inner_roi_y, outer_roi_low_x, outer_roi_low_y, outer_roi_high_x, outer_roi_high_y, outer_roi_lim_x, outer_roi_lim_y

def allsky_map(m, title, vmin, vmax, xsize = 2000):
    '''
    Plot (log) all-sky map with healpy
    Example: https://healpy.readthedocs.io/en/latest/newvisufunc_example.html
    Source: v0.3/notebooks_canon/some_figures.ipynb

    Parameters
    ----------
    m : np.ndarray
        Healpix map
    title : str
        Title of the plot
    vmin : float
        Minimum value of the colorbar
    vmax : float
        Maximum value of the colorbar
    '''

    # set custom tick labels
    pre_xtick_labels = [-120, -60, 0, 60, 120]
    xtick_labels = [str(i) + '$^\circ$' for i in pre_xtick_labels]
    pre_ytick_labels = [-60, -30, 0, 30, 60]
    ytick_labels = [str(i) + '$^\circ$' for i in pre_ytick_labels]

    projview(
        m, coord=["C"], flip = "astro", projection_type="mollweide", title = title,
        xlabel = '$\ell$', ylabel = '$b$', 
        latitude_grid_spacing = 30, custom_xtick_labels=xtick_labels, custom_ytick_labels=ytick_labels,
        graticule = True, graticule_labels = True, unit='$\log_{10}(\lambda)$',
        xsize = xsize, min = vmin, max = vmax,
        # override_plot_properties = {'figure_width' : 8},
        )

    # load boundary shapes
    l_list, b_list, _, _, _, _, _, _, _, _ = preprocess_map_shapes()
    plt.plot(np.deg2rad(l_list), np.deg2rad(b_list), color="k", ls = "-", lw = 1.)

def inner_roi_map(m, title, vmin, vmax, subplot=None, display_x_info = True, display_y_info=True, cbar = True, cb_orientation = 'horizontal', xsize = 2000, unit = '$\\log_{10}(\lambda)$'):
    '''
    Plot map in our inner ROI with healpy
    Example: https://healpy.readthedocs.io/en/latest/newvisufunc_example.html
    Source: v0.3/notebooks_canon/some_figures.ipynb

    Parameters
    ----------
    m : np.ndarray
        Healpix map
    title : str
        Title of the plot
    vmin : float
        Minimum value of the colorbar
    vmax : float
        Maximum value of the colorbar
    '''

    # set custom tick labels
    pre_xtick_labels = ['dummy', -20, -10, 0, 10, 20]
    xtick_labels = [str(i) + '$^\circ$' for i in pre_xtick_labels]
    pre_ytick_labels = ['dummy', -20, -10, 0, 10, 20]
    ytick_labels = [str(i) + '$^\circ$' for i in pre_ytick_labels]

    projview(
        m,
        coord=["G"], 
        flip = "astro", 
        projection_type="cart", 
        title = title,
        xlabel = '$\ell$' if display_x_info else None,
        ylabel = '$b$' if display_y_info else None, 
        xsize = xsize, # 
        latitude_grid_spacing = 10,
        longitude_grid_spacing = 10, 
        custom_xtick_labels=xtick_labels if display_x_info else ['' for i in pre_xtick_labels],
        custom_ytick_labels=ytick_labels if display_y_info else ['' for i in pre_ytick_labels],
        graticule = True, 
        graticule_labels = True, 
        unit=unit,
        cbar = cbar,
        min = vmin, max = vmax,
        cb_orientation = cb_orientation, 
        override_plot_properties = {'cbar_pad': 0.1, 'cbar_tick_direction' : 'out'},
        hold = True, # needed to plot shapes over map
        sub = subplot,
        )
    
    # load boundary and fill shapes
    l_list, b_list, inner_roi_x, inner_roi_y, outer_roi_low_x, outer_roi_low_y, outer_roi_high_x, outer_roi_high_y, outer_roi_lim_x, outer_roi_lim_y = preprocess_map_shapes()

    plt.xlim(np.deg2rad(-20),np.deg2rad(20))
    plt.ylim(np.deg2rad(-20),np.deg2rad(20))
    plt.plot(np.deg2rad(l_list), np.deg2rad(b_list), color="r", ls = "-", lw = 1.)

    plt.xlim(np.deg2rad(-20),np.deg2rad(20))
    plt.ylim(np.deg2rad(-20),np.deg2rad(20))
    plt.plot(np.deg2rad(l_list), np.deg2rad(b_list), color="k", ls = "-", lw = 1.)

    plt.plot(inner_roi_x, inner_roi_y, color="k", ls = "-", lw = 1.)
    plt.plot(outer_roi_low_x, outer_roi_low_y, color="k", ls = "-", lw = 1.)
    plt.plot(outer_roi_high_x, outer_roi_high_y, color="k", ls = "-", lw = 1.)

    annulus_roi_x = [inner_roi_x, outer_roi_low_x[::-1]]
    annulus_roi_y = [inner_roi_y, outer_roi_low_y[::-1]]

    plt.fill(np.ravel(annulus_roi_x), np.ravel(annulus_roi_y), color = 'gray')
    plt.grid(False)

def complete_roi_map(m, title, vmin, vmax, subplot = 111, display_x_info = True, display_y_info = True):
    '''
    Plot map in our inner ROI with healpy
    Example: https://healpy.readthedocs.io/en/latest/newvisufunc_example.html
    Source: v0.3/notebooks_canon/some_figures.ipynb

    Parameters
    ----------
    m : np.ndarray
        Healpix map
    figdir : str
        Path to save figure
    title : str
        Title of the plot
    vmin : float
        Minimum value of the colorbar
    vmax : float
        Maximum value of the colorbar
    '''

    # set custom tick labels
    pre_xtick_labels = ['dummy', -40., -20., 0., 20., 40.]
    xtick_labels = [str(i) + '$^\circ$' for i in pre_xtick_labels]
    pre_ytick_labels = ['dummy', -40., -20., 0., 20., 40.]
    ytick_labels = [str(i) + '$^\circ$' for i in pre_ytick_labels]

    # generate map
    projview(
        m,
        coord=["G"], 
        flip = "astro", 
        projection_type="cart", 
        title = title,
        xlabel = '$\ell$' if display_x_info else None, 
        ylabel = '$b$' if display_y_info else None, 
        xsize = 2000,
        latitude_grid_spacing = 20,
        longitude_grid_spacing = 20, 
        custom_xtick_labels=xtick_labels if display_x_info else ['' for i in pre_xtick_labels],
        custom_ytick_labels=ytick_labels if display_y_info else ['' for i in pre_ytick_labels],
        graticule = True, 
        graticule_labels = True, 
        unit='$\\log_{10}(\lambda)$',
        min = 0, max = 2.4,
        cb_orientation = 'horizontal', 
        override_plot_properties = {'cbar_pad': 0.1},
        hold = True,
        sub = subplot,
        )
    
    l_list, b_list, inner_roi_x, inner_roi_y, outer_roi_low_x, outer_roi_low_y, outer_roi_high_x, outer_roi_high_y, outer_roi_lim_x, outer_roi_lim_y = preprocess_map_shapes()

    plt.xlim(np.deg2rad(-40),np.deg2rad(40))
    plt.ylim(np.deg2rad(-40),np.deg2rad(40))
    plt.plot(np.deg2rad(l_list), np.deg2rad(b_list), color="r", ls = "-", lw = 1.)

    plt.xlim(np.deg2rad(-40),np.deg2rad(40))
    plt.ylim(np.deg2rad(-40),np.deg2rad(40))
    plt.plot(np.deg2rad(l_list), np.deg2rad(b_list), color="k", ls = "-", lw = 1.)

    plt.plot(inner_roi_x, inner_roi_y, color="k", ls = "-", lw = 1.)
    plt.plot(outer_roi_low_x, outer_roi_low_y, color="k", ls = "-", lw = 1.)
    plt.plot(outer_roi_high_x, outer_roi_high_y, color="k", ls = "-", lw = 1.)

    annulus_roi_x = [inner_roi_x, outer_roi_low_x[::-1]]
    annulus_roi_y = [inner_roi_y, outer_roi_low_y[::-1]]

    plt.fill(np.ravel(annulus_roi_x), np.ravel(annulus_roi_y), color = 'gray')

    annulus_roi_x = [outer_roi_high_x, outer_roi_lim_x[::-1]]
    annulus_roi_y = [outer_roi_high_y, outer_roi_lim_y[::-1]]

    plt.fill(np.ravel(annulus_roi_x), np.ravel(annulus_roi_y), color = 'gray')

    # inside inner_roi, add text saying "GP + Templates"
    plt.text(0, 0, '${\\rm \\bf{Inner \\ ROI}}$', horizontalalignment='center', verticalalignment='center', weight = 'bold')

    # inside buffer, add text "No Fitting"
    plt.text(0, 0.42, '${\\rm \\bf{Buffer}}$', horizontalalignment='center', verticalalignment='center', weight = 'bold')

    # inside buffer, add text "No Fitting"
    plt.text(0, 0.59, '${\\rm \\bf{Outer \\ ROI}}$', horizontalalignment='center', verticalalignment='center', color = 'black')

    plt.grid(False)


def outer_roi_map(m, title, vmin, vmax, subplot = 111, display_x_info = True, display_y_info = True):
    '''
    Plot map in our inner ROI with healpy
    Example: https://healpy.readthedocs.io/en/latest/newvisufunc_example.html
    Source: v0.3/notebooks_canon/some_figures.ipynb

    Parameters
    ----------
    m : np.ndarray
        Healpix map
    figdir : str
        Path to save figure
    title : str
        Title of the plot
    vmin : float
        Minimum value of the colorbar
    vmax : float
        Maximum value of the colorbar
    '''

    # set custom tick labels
    pre_xtick_labels = ['dummy', -40., -20., 0., 20., 40.]
    xtick_labels = [str(i) + '$^\circ$' for i in pre_xtick_labels]
    pre_ytick_labels = ['dummy', -40., -20., 0., 20., 40.]
    ytick_labels = [str(i) + '$^\circ$' for i in pre_ytick_labels]

    # generate map
    projview(
        m,
        coord=["G"], 
        flip = "astro", 
        projection_type="cart", 
        title = title,
        xlabel = '$\ell$' if display_x_info else None, 
        ylabel = '$b$' if display_y_info else None, 
        xsize = 2000,
        latitude_grid_spacing = 20,
        longitude_grid_spacing = 20, 
        custom_xtick_labels=xtick_labels if display_x_info else ['' for i in pre_xtick_labels],
        custom_ytick_labels=ytick_labels if display_y_info else ['' for i in pre_ytick_labels],
        graticule = True, 
        graticule_labels = True, 
        unit='$\\log_{10}(\lambda)$',
        min = vmin, max = vmax,
        cb_orientation = 'horizontal', 
        override_plot_properties = {'cbar_pad': 0.1},
        hold = True,
        sub = subplot,
        )
    
    l_list, b_list, inner_roi_x, inner_roi_y, outer_roi_low_x, outer_roi_low_y, outer_roi_high_x, outer_roi_high_y, outer_roi_lim_x, outer_roi_lim_y = preprocess_map_shapes()

    plt.xlim(np.deg2rad(-40),np.deg2rad(40))
    plt.ylim(np.deg2rad(-40),np.deg2rad(40))

    plt.plot(outer_roi_high_x, outer_roi_high_y, color="k", ls = "-", lw = 1.)

    annulus_roi_x = [outer_roi_high_x, outer_roi_lim_x[::-1]]
    annulus_roi_y = [outer_roi_high_y, outer_roi_lim_y[::-1]]

    plt.fill(np.ravel(annulus_roi_x), np.ravel(annulus_roi_y), color = 'gray')

    plt.grid(False)

def bub_map(m, title, subplot = 111, display_x_info = True, display_y_info = True):
    # set custom tick labels
    pre_xtick_labels = ['dummy', -40., -20., 0., 20., 40.]
    xtick_labels = [str(i) + '$^\circ$' for i in pre_xtick_labels]
    pre_ytick_labels = ['dummy', -40., -20., 0., 20., 40.]
    ytick_labels = [str(i) + '$^\circ$' for i in pre_ytick_labels]

    # generate map
    projview(
        m,
        coord=["G"], 
        flip = "astro", 
        projection_type="cart", 
        title = title,
        xlabel = '$\ell$' if display_x_info else None, 
        ylabel = '$b$' if display_y_info else None, 
        xsize = 2000,
        latitude_grid_spacing = 20,
        longitude_grid_spacing = 20, 
        custom_xtick_labels=xtick_labels if display_x_info else ['' for i in pre_xtick_labels],
        custom_ytick_labels=ytick_labels if display_y_info else ['' for i in pre_ytick_labels],
        graticule = True, 
        graticule_labels = True, 
        unit='$\\lambda$',
        cb_orientation = 'horizontal', 
        override_plot_properties = {'cbar_pad': 0.1},
        hold = True,
        sub = subplot,
        )
    
    
    plt.xlim(np.deg2rad(-40),np.deg2rad(40))
    plt.ylim(np.deg2rad(-40),np.deg2rad(40))

    plt.grid(False)

def add_gray_region():
    # load boundary and fill shapes
    l_list, b_list, inner_roi_x, inner_roi_y, outer_roi_low_x, outer_roi_low_y, outer_roi_high_x, outer_roi_high_y, outer_roi_lim_x, outer_roi_lim_y = preprocess_map_shapes()

    plt.xlim(np.deg2rad(-20),np.deg2rad(20))
    plt.ylim(np.deg2rad(-20),np.deg2rad(20))
    plt.plot(np.deg2rad(l_list), np.deg2rad(b_list), color="r", ls = "-", lw = 1.)

    plt.xlim(np.deg2rad(-20),np.deg2rad(20))
    plt.ylim(np.deg2rad(-20),np.deg2rad(20))
    plt.plot(np.deg2rad(l_list), np.deg2rad(b_list), color="k", ls = "-", lw = 1.)

    plt.plot(inner_roi_x, inner_roi_y, color="k", ls = "-", lw = 1.)
    plt.plot(outer_roi_low_x, outer_roi_low_y, color="k", ls = "-", lw = 1.)
    plt.plot(outer_roi_high_x, outer_roi_high_y, color="k", ls = "-", lw = 1.)

    annulus_roi_x = [inner_roi_x, outer_roi_low_x[::-1]]
    annulus_roi_y = [inner_roi_y, outer_roi_low_y[::-1]]

    plt.fill(np.ravel(annulus_roi_x), np.ravel(annulus_roi_y), color = 'gray')
    

def cart_plot_1d(q, 
                 sim_cart=None, raw_cart=None,
                 mask_map_cart = None,
                 n_pixels = 160, res_scale = 1, map_size = 40,
                 slice_dir = 'horizontal', slice_val = 2., 
                 yscale = 'linear', ylim = None,
                 q_color = 'red', line_color = 'blue', scatter_color = 'k', ylabel = 'Counts', ls = '-',
                 samples = None):
    '''
    Assumes map is 40 deg x 40 deg (similar to ed_fcts_amarel.cart_coords standard settings)

    Currently only supports horizontal slices. For vertical and diagonal slices,
    see cart_plot_1d in ed_fcts_amarel.py for inspiration.
    '''
    # generate cartesian grid
    Nx1, Nx2, x1_plt, x2_plt, x1_c, x2_c, x = ef.cart_coords(n_pixels, res_scale, map_size)
    pix_scale = map_size / n_pixels

    if slice_dir == 'horizontal':
        y_slice = slice_val
        ny = np.where(np.abs(x2_c - y_slice) < pix_scale * res_scale)[0][1]
        print('Slice at y = {:.5f} deg'.format(x[ny,0,1]))

        plt.plot(x[ny,:,0], q[2][ny,:], c = q_color, label = 'Median')
        plt.fill_between(x[ny,:,0], q[1][ny,:], q[3][ny,:], color = q_color, alpha = 0.2, label = '68$\%$')
        plt.fill_between(x[ny,:,0], q[0][ny,:], q[4][ny,:], color = q_color, alpha = 0.1, label = '95$\%$')
        if sim_cart is not None:
            plt.plot(x[ny,:,0], sim_cart[ny,:], c = line_color, label = 'True', ls = ls)
        if raw_cart is not None:
            plt.errorbar(x[ny,:,0], raw_cart[ny,:], fmt = 'o', c = scatter_color, alpha = 0.5, label = 'Data')
        plt.xlabel('$\ell$ (deg)')
        plt.ylabel(ylabel)
        plt.legend(frameon = False, fontsize = 14)
        plt.axvline(0, color='k', ls = '--', lw = 0.5)
        plt.yscale(yscale)

        if samples is not None:
            for i in range(4):
                plt.plot(x[ny,:,0], samples[i,ny,:], c = 'gray', alpha = 0.3)

        if mask_map_cart is not None:
            mask_map_cart_slice = mask_map_cart[ny,:]

            # fill points where mask_map_cart is nan
            nan_mask = np.isnan(mask_map_cart_slice)
            x_nan = x[ny,nan_mask,0]
            y_nan = np.zeros_like(x_nan)
            # ax.scatter(x_nan, y_nan, c = 'k', s = 1)

            # fill between points in x_nan that are separated by at most 0.5 deg
            x_nan_diff = np.diff(x_nan)

            # find the indices of the x_nan_diff that are greater than 0.25 in order to 
            # partition x_nan into separate arrays with members that are less than 0.25 separated
            # added 0.01 to account for floating point error

            split_indices = np.where(x_nan_diff > pix_scale + 0.01)[0] + 1 # 1 is added to account for the diff shift in indices
            split_indices = np.insert(split_indices, 0, 0)
            split_indices = np.append(split_indices, len(x_nan))

            for i in range(len(split_indices) - 1):
                x_fill = x_nan[split_indices[i]:split_indices[i+1]]
                y_fill = np.zeros_like(x_fill) + np.min([0, np.min(q[0][ny,:])]) - 0.25 - 1 # complicated min expression in case q negative or positive
                plt.fill_between(x_fill, 100 * y_fill, 100 * np.max(q[-1][ny,:]) + 1, color = 'gray', alpha = 0.175, edgecolor = None)

            plt.xlim(-20,20)
            if ylim is None:
                plt.ylim(np.min([-0.25, np.min(q[0][ny,:]) - 0.25]), np.max(q[-1][ny,:]) + 0.25) # complicated min expression in case q negative or positive
            else:
                plt.ylim(ylim)
    elif slice_dir == 'vertical':
        x_slice = slice_val
        nx = np.where(np.abs(x1_c - x_slice) < pix_scale * res_scale)[0][1]
        print('Slice at x = {:.5f} deg'.format(x[0,nx,0]))

        plt.plot(x[:,nx,1], q[2][:,nx], c = q_color, label = 'Median')
        plt.fill_between(x[:,nx,1], q[1][:,nx], q[3][:,nx], color = q_color, alpha = 0.2, label = '68$\%$')
        plt.fill_between(x[:,nx,1], q[0][:,nx], q[4][:,nx], color = q_color, alpha = 0.1, label = '95$\%$')
        if sim_cart is not None:
            plt.plot(x[:,nx,1], sim_cart[:,nx], c = line_color, label = 'True', ls = ls)
        if raw_cart is not None:
            plt.errorbar(x[:,nx,1], raw_cart[:,nx], fmt = 'o', c = scatter_color, alpha = 0.5, label = 'Data')
        plt.xlabel('b (deg)')
        plt.ylabel(ylabel)
        plt.legend(frameon = False, fontsize = 14)
        plt.axvline(0, color='k', ls = '--', lw = 0.5)
        plt.yscale(yscale)

        if samples is not None:
            for i in range(4):
                plt.plot(x[:,nx,1], samples[i,:,nx], c = 'gray', alpha = 0.3)

        if mask_map_cart is not None:
            mask_map_cart_slice = mask_map_cart[:,nx]

            # fill points where mask_map_cart is nan
            nan_mask = np.isnan(mask_map_cart_slice)
            x_nan = x[nan_mask,nx,1]
            y_nan = np.zeros_like(x_nan)
            # ax.scatter(x_nan, y_nan, c = 'k', s = 1)

            # fill between points in x_nan that are separated by at most 0.5 deg
            x_nan_diff = np.diff(x_nan)

            # find the indices of the x_nan_diff that are greater than 0.25 in order to 
            # partition x_nan into separate arrays with members that are less than 0.25 separated
            # added 0.01 to account for floating point error

            split_indices = np.where(x_nan_diff > pix_scale + 0.01)[0] + 1 # 1 is added to account for the diff shift in indices
            split_indices = np.insert(split_indices, 0, 0)
            split_indices = np.append(split_indices, len(x_nan))

            for i in range(len(split_indices) - 1):
                x_fill = x_nan[split_indices[i]:split_indices[i+1]]
                y_fill = np.zeros_like(x_fill) + np.min([0, np.min(q[0][:,nx])]) - 0.25 - 1 # complicated min expression in case q negative or positive
                plt.fill_between(x_fill, 100 * y_fill, 100 * np.max(q[-1][:,nx]) + 1, color = 'gray', alpha = 0.175, edgecolor = None)

            plt.xlim(-20,20)
            if ylim is None:
                plt.ylim(np.min([-0.25, np.min(q[0][:,nx]) - 0.25]), np.max(q[-1][:,nx]) + 0.25) # complicated min expression in case q negative or positive
            else:
                plt.ylim(ylim)
        else:
            print('Only horizontal slices are supported at the moment.')

def cart_plot_1d_multi(q_list, 
                 sim_cart=None, raw_cart=None,
                 mask_map_cart = None,
                 n_pixels = 160, res_scale = 1, map_size = 40,
                 slice_dir = 'horizontal', slice_val = 2., 
                 yscale = 'linear', ylim = None,
                 q_colors = None, line_color = 'blue', scatter_color = 'k', ylabel = 'Counts', ls = '-', q_labels = None,
                 samples = None):
    '''
    Assumes map is 40 deg x 40 deg (similar to ed_fcts_amarel.cart_coords standard settings)

    Currently only supports horizontal slices. For vertical and diagonal slices,
    see cart_plot_1d in ed_fcts_amarel.py for inspiration.
    '''
    # generate cartesian grid
    Nx1, Nx2, x1_plt, x2_plt, x1_c, x2_c, x = ef.cart_coords(n_pixels, res_scale, map_size)
    pix_scale = map_size / n_pixels


    if slice_dir == 'horizontal':
        y_slice = slice_val
        ny = np.where(np.abs(x2_c - y_slice) < pix_scale * res_scale)[0][1]
        print('Slice at y = {:.5f} deg'.format(x[ny,0,1]))

        for i, q in enumerate(q_list):
            q_label = q_labels[i]
            q_color = q_colors[i]
            plt.plot(x[ny,:,0], q[1][ny,:], c = q_color, label = q_label)
            plt.fill_between(x[ny,:,0], q[0][ny,:], q[2][ny,:], color = q_color, alpha = 0.2)
        if sim_cart is not None:
            plt.plot(x[ny,:,0], sim_cart[ny,:], c = line_color, label = 'True', ls = ls)
        if raw_cart is not None:
            plt.errorbar(x[ny,:,0], raw_cart[ny,:], fmt = 'o', c = scatter_color, alpha = 0.5, label = 'Data')
        plt.xlabel('$\ell$ (deg)')
        plt.ylabel(ylabel)
        plt.legend(frameon = False, fontsize = 14)
        plt.axvline(0, color='k', ls = '--', lw = 0.5)
        plt.yscale(yscale)

        if samples is not None:
            for i in range(4):
                plt.plot(x[ny,:,0], samples[i,ny,:], c = 'gray', alpha = 0.3)

        if mask_map_cart is not None:
            mask_map_cart_slice = mask_map_cart[ny,:]

            # fill points where mask_map_cart is nan
            nan_mask = np.isnan(mask_map_cart_slice)
            x_nan = x[ny,nan_mask,0]
            y_nan = np.zeros_like(x_nan)
            # ax.scatter(x_nan, y_nan, c = 'k', s = 1)

            # fill between points in x_nan that are separated by at most 0.5 deg
            x_nan_diff = np.diff(x_nan)

            # find the indices of the x_nan_diff that are greater than 0.25 in order to 
            # partition x_nan into separate arrays with members that are less than 0.25 separated
            # added 0.01 to account for floating point error

            split_indices = np.where(x_nan_diff > pix_scale + 0.01)[0] + 1 # 1 is added to account for the diff shift in indices
            split_indices = np.insert(split_indices, 0, 0)
            split_indices = np.append(split_indices, len(x_nan))

            # find max q from q_list
            max_q = np.max([np.max(q[-1][ny,:]) for q in q_list])
            for i in range(len(split_indices) - 1):
                x_fill = x_nan[split_indices[i]:split_indices[i+1]]
                y_fill = np.zeros_like(x_fill) + np.min([0, np.min(q[0][ny,:])]) - 0.25 - 1 # complicated min expression in case q negative or positive
                plt.fill_between(x_fill, 100 * y_fill, 100 * max_q + 1, color = 'gray', alpha = 0.175, edgecolor = None)

            plt.xlim(-20,20)
            if ylim is None:
                plt.ylim(np.min([-0.25, np.min(q[0][ny,:]) - 0.25]), max_q + 0.25) # complicated min expression in case q negative or positive
            else:
                plt.ylim(ylim)
    elif slice_dir == 'vertical':
        x_slice = slice_val
        nx = np.where(np.abs(x1_c - x_slice) < pix_scale * res_scale)[0][1]
        print('Slice at x = {:.5f} deg'.format(x[0,nx,0]))

        for i, q in enumerate(q_list):
            q_label = q_labels[i]
            q_color = q_colors[i]
            plt.plot(x[:,nx,1], q[1][:,nx], c = q_color, label = q_label)
            plt.fill_between(x[:,nx,1], q[0][:,nx], q[2][:,nx], color = q_color, alpha = 0.2)
        if sim_cart is not None:
            plt.plot(x[:,nx,1], sim_cart[:,nx], c = line_color, label = 'True', ls = ls)
        if raw_cart is not None:
            plt.errorbar(x[:,nx,1], raw_cart[:,nx], fmt = 'o', c = scatter_color, alpha = 0.5, label = 'Data')
        plt.xlabel('b (deg)')
        plt.ylabel(ylabel)
        plt.legend(frameon = False, fontsize = 14)
        plt.axvline(0, color='k', ls = '--', lw = 0.5)
        plt.yscale(yscale)

        if samples is not None:
            for i in range(4):
                plt.plot(x[:,nx,1], samples[i,:,nx], c = 'gray', alpha = 0.3)

        if mask_map_cart is not None:
            mask_map_cart_slice = mask_map_cart[:,nx]

            # fill points where mask_map_cart is nan
            nan_mask = np.isnan(mask_map_cart_slice)
            x_nan = x[nan_mask,nx,1]
            y_nan = np.zeros_like(x_nan)
            # ax.scatter(x_nan, y_nan, c = 'k', s = 1)

            # fill between points in x_nan that are separated by at most 0.5 deg
            x_nan_diff = np.diff(x_nan)

            # find the indices of the x_nan_diff that are greater than 0.25 in order to 
            # partition x_nan into separate arrays with members that are less than 0.25 separated
            # added 0.01 to account for floating point error

            split_indices = np.where(x_nan_diff > pix_scale + 0.01)[0] + 1 # 1 is added to account for the diff shift in indices
            split_indices = np.insert(split_indices, 0, 0)
            split_indices = np.append(split_indices, len(x_nan))

            # find max q from q_list
            max_q = np.max([np.max(q[-1][:,nx]) for q in q_list])
            for i in range(len(split_indices) - 1):
                x_fill = x_nan[split_indices[i]:split_indices[i+1]]
                y_fill = np.zeros_like(x_fill) + np.min([0, np.min(q[0][:,nx])]) - 0.25 - 1 # complicated min expression in case q negative or positive
                plt.fill_between(x_fill, 100 * y_fill, 100 * max_q + 1, color = 'gray', alpha = 0.175, edgecolor = None)

            plt.xlim(-20,20)
            if ylim is None:
                plt.ylim(np.min([-0.25, np.min(q[0][:,nx]) - 0.25]), max_q + 0.25) # complicated min expression in case q negative or positive
            else:
                plt.ylim(ylim)
        else:
            print('Only horizontal slices are supported at the moment.')

def violin_plot(all_data, colors, ax):
    '''
    Parameters
    ----------
    all_data : list
        List of arrays of data to plot (in our case, ll samples)
    colors : list
        List of colors for each data array
    ax : matplotlib.axes.Axes
        Axes
    '''
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # load data
    # all_data = [ll_total - ll_total.mean(), ll_inner - ll_inner.mean(), ll_outer - ll_outer.mean()]
    # colors = ['red', 'blue', 'green']

    # plot violin plot
    plots = ax.violinplot(all_data,
                    vert = False,
                    showmeans=False,
                    showmedians=True)

    # Set the color of the violin patches
    for pc, color in zip(plots['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)

    for partname in ('cbars','cmins','cmaxes','cmedians'):
        plots[partname].set_color(colors)

    # Set the color of the median lines
    plots['cmedians'].set_colors(colors)

    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value


    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')

    quartile1, medians, quartile3 = np.percentile(all_data, [16, 50, 84], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(all_data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(medians, inds, marker='o', color='white', s=30, zorder=3)
    ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # repeat this for each hline and whisker
    for i in range(len(inds)):
        ax.hlines(inds[i], quartile1[i], quartile3[i], color=colors[i], linestyle='-', lw=5)
        ax.hlines(inds[i], whiskers_min[i], whiskers_max[i], color=colors[i], linestyle='-', lw=1)

    # adding horizontal grid lines

    ax.yaxis.grid(True)
    ax.set_yticks([y + 1 for y in range(len(all_data))],
                    labels=['Total', 'Inner', 'Outer'])
    ax.set_xlabel('$\\Delta LL$')
    ax.set_ylabel('ROI')
    # ax.set_xlim([-5,5])
    ax.axvline(0, color='black', lw=1, ls='--')

def rel_norm_comparisons_all(q, include_model_o = False):
    '''
    Parameters
    ----------
    q : tuple
        Tuple of arrays containing 1\sigma quantiles of relative normalization

    '''
    # highlight best models (dif_names ordered from 0 - 79)
    roman_nums_arr = [ef.int_to_Roman(i) for i in range(1,80+1)]
    best_models_rom = ['X', 'XV', 'XLVIII', 'XLIX', 'LIII']
    best_models_idx = [roman_nums_arr.index(r) for r in best_models_rom]

    if include_model_o == False:
        x = np.arange(1,80+1)
        c = ['blue' for i in range(80)]
        best_models = []
        for i in best_models_idx:
            c[i] = 'red'

        for i in range(80):
            plt.errorbar(x[i], q[1][i], yerr= np.array([[q[1][i] - q[0][i]], [q[2][i] - q[1][i]]]), fmt='o', color=c[i])
    else:
        x = np.arange(0,80+1)
        c = ['blue' for i in range(80)]
        best_models = []
        for i in best_models_idx:
            c[i] = 'red'
        c = ['green'] + c # add green for gceNNo
        for i in range(80+1):
            plt.errorbar(x[i], q[1][i], yerr= np.array([[q[1][i] - q[0][i]], [q[2][i] - q[1][i]]]), fmt='o', color=c[i])

def scan_plot(x_list, q_list, xlabel, ordered_names, all_temp_names, ccodes):
    Ns = len(x_list)
    for k in range(len(ordered_names)):
        name = ordered_names[k]
        idx = all_temp_names.index(name)
        ccode = ccodes[idx]
        low_list = [q_list[n][name][0] for n in range(Ns)]
        mean_list = [q_list[n][name][1] for n in range(Ns)]
        high_list = [q_list[n][name][2] for n in range(Ns)]
        plt.plot(x_list, mean_list, label = name, color = ccode)
        plt.fill_between(x_list, low_list, high_list, alpha = 0.2, color = ccode)
    plt.xlabel(xlabel)
    plt.ylabel('$\\frac{(\\lambda- \\lambda_{true})}{\\lambda_{true}}$')
    plt.axhline(0, linestyle = '--', color = 'k')
    plt.legend()
    plt.xlim(min(x_list), max(x_list))

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used