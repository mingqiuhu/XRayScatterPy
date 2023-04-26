# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:28:22 2022

@author: pkuhu
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import reflectivity_pltlib


class Model:
    '''1D numerical model of the scattering length density as a function of
    distance from the top surface.

    layer_pts: number of points to be used to disretize the layered structure

    thick: 1D array of the thickness of individule layers

    msr: 1D array of mean square roughness of individule layers

    sldr: 1D array of the real scattering length density of individual layers

    sldi: 1D array of the imaginary scattering length density

    sum_thick: total thickness of the layered structure

    prof_thick: 1D array of the thickness of equally thick discretized layers

    prof_sldr: 1D array of the real SLD of equally thick discretized layers

    prof_sldi: 1D array of the imaginary SLD of discretized layers

    discretize(): return a new object consisting the discretized layers

    plt_ideal(): plot the ideal real and imaginary SLD as a function of depth

    plt_broadened(): plot the real and imaginary SLD as a function of depth
    after considering the interfacial roughness

    plt_all(): plot the real and imaginary SLD as a function of depth with or
    without considering the interfacial roughness
    '''

    def __init__(self, creat_prof: bool, layer_pts: int, *params):

        self.layer_pts = layer_pts  # number of layers after discretization
        params = np.array(params).reshape(-1, 4)  # from top to bottom

        if params.shape[0] < 2:
            sys.exit('input error: at least 2 layers required')

        self.thick = params[:, 0]  # from top to bottom

        self.msr = params[:, 1]  # from top to bottom
        if self.msr[0] != 0:
            sys.exit('input error: msr of top layer must be 0')
        elif np.any(self.msr[2:] > self.thick[1:-1] / 8):
            sys.exit('input error: msr must be smaller than 1/8 d')
        elif np.any(self.msr[1:-1] > self.thick[1:-1] / 8):
            sys.exit('input error: msr must be smaller than 1/8 d')

        self.sldr = params[:, 2]  # from top to bottom
        self.sldi = params[:, 3]  # from top to bottom
        if self.sldr[0] != 0 or self.sldi[0] != 0:
            sys.exit('input error: sld of top layer must be 0')

        self.thick[0] = 5 * self.msr[1]
        self.thick[-1] = 5 * self.msr[-1]
        # Thickness of bottom and top layer doesn't matter for the calculation.
        # This value is chosen to fully display the broadening.

        self.sum_thick = np.sum(self.thick)

        if creat_prof:
            (self.prof_thick,
             self._prof_sldr_ideal,
             self._prof_sldi_ideal) = self._calc_prof_ideal()
            # The thickness of each point in the new profile is total thickness
            # divided by n_layers.

            (self.prof_sldr, self.prof_sldi) = self._calc_prof_broadened()

    def _func_i_splt(self, cumsum_thick):
        '''This function takes a 1D array of cummulatively summed thickness and
        returns a 1D array of indexes beginning with the index of the first
        cutting point and ending at index = layer_pts. After cutting, there
        will be layer_pts+1 subarrays. The last array is empty and shoud be
        stripped.'''

        return (cumsum_thick / self.sum_thick * self.layer_pts).astype(int)

    def _calc_prof_ideal(self):
        '''This function calculates the scattering length density as a function
        of the distance from the top surface, assuming sharp interfaces. It
        returns a 1D array of the distance from surface (prof_thick), and the
        real and imaginary SLD from surface, prof_sldr_ideal and
        prof_sldi_ideal'''

        prof_thick = np.linspace(0, self.sum_thick, self.layer_pts + 1)[:-1]
        # This is to ensure there are total layer_pts points and the thickness
        # represented by each point is sum_thick/layer_pts

        i_splt_layer = self._func_i_splt(np.cumsum(self.thick))
        # Obtain the index where split should occur.

        zeros_splt_layer = np.split(np.zeros(self.layer_pts), i_splt_layer)
        zeros_splt_layer = np.array(zeros_splt_layer, dtype='object')[:-1]
        # Array of zeros after splitting. Each element is an array with one sld

        prof_sldr_ideal = np.hstack(zeros_splt_layer + self.sldr)
        prof_sldi_ideal = np.hstack(zeros_splt_layer + self.sldi)

        return prof_thick, prof_sldr_ideal, prof_sldi_ideal

    def _calc_prof_broadened(self):
        '''This function calculates the continous scattering density as a
        function of distance from the top surface, considering the roughness of
        each interface. It returns the real and imaginary scattering length
        density, prof_sldr and prof_sldi, from the top surface.'''

        gf1d = np.vectorize(scipy.ndimage.gaussian_filter1d, otypes=[object])
        # This is later used when msr of each interface needs to be broadcasted
        # to each discretized layer corresponding to that interface.

        n_msr = self.msr[1:] / self.sum_thick * self.layer_pts
        # This converts the unit of msr from real space A to pts.

        i_splt_inter = self._func_i_splt(
            np.cumsum(self.thick) - self.thick / 2)
        # index for splitting in the middle of each layer. The first and last
        # layer need not to be broadened.

        def prof_splt(arr):
            return np.array(np.split(arr, i_splt_inter), dtype='object')
        # This function splits the profiles in the middle of each layer.

        prof_splt_sldr = prof_splt(self._prof_sldr_ideal)
        prof_splt_sldi = prof_splt(self._prof_sldi_ideal)
        prof_splt_sldr[1:-1] = gf1d(prof_splt_sldr[1:-1],
                                    n_msr, mode='nearest')
        prof_splt_sldi[1:-1] = gf1d(prof_splt_sldi[1:-1],
                                    n_msr, mode='nearest')
        prof_sldr = np.hstack(prof_splt_sldr[:])
        prof_sldi = np.hstack(prof_splt_sldi[:])
        # Concanenate the arrays after broadening.

        return prof_sldr, prof_sldi

    def plt_ideal(self):
        '''plot ideal SLD as a function of distance from the surface'''

        reflectivity_pltlib.plt_set_params_before()
        plt.plot(self.prof_thick, self._prof_sldr_ideal * 1e+6, label='SLD')
        plt.plot(self.prof_thick, self._prof_sldi_ideal * 1e+6, label='SLDi')
        reflectivity_pltlib.plt_set_params_after_sld()

    def plt_broadened(self):
        '''plot broadened SLD as a function of distance from the surface'''

        reflectivity_pltlib.plt_set_params_before()
        plt.plot(self.prof_thick, self.prof_sldr * 1e+6, label='Diffused SLD')
        plt.plot(self.prof_thick, self.prof_sldi * 1e+6, label='Diffused SLDi')
        reflectivity_pltlib.plt_set_params_after_sld()

    def plt_all(self):
        '''plot all SLD as a function of distance from the surface'''

        reflectivity_pltlib.plt_set_params_before()
        plt.plot(self.prof_thick, self._prof_sldr_ideal * 1e+6, label='SLD')
        plt.plot(self.prof_thick, self._prof_sldi_ideal * 1e+6, label='SLDi')
        plt.plot(self.prof_thick, self.prof_sldr * 1e+6, label='Diffused SLD')
        plt.plot(self.prof_thick, self.prof_sldi * 1e+6, label='Diffused SLDi')
        reflectivity_pltlib.plt_set_params_after_sld()

    def discretize(self):
        '''This function discretizes a multi-layer model with diffuse
        interfaces into individual thin layers with sharp interfaces. It
        returns a new object containing the discretized layers. '''

        new_params = np.dstack((np.sum(self.thick) / self.layer_pts *
                                np.ones(self.layer_pts),
                                np.zeros(self.layer_pts),  # roughness = 0
                                self.prof_sldr,  # Discretized sldr
                                self.prof_sldi))  # Discretized sldi
        new_model = Model(False, self.layer_pts, new_params.flatten())

        return new_model
