# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:01:30 2022

@author: pkuhu
"""

import time
import numpy as np
# from numba import jit
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from reflectivity_model import Model


class Calculator():
    '''This is the base class of all calculators.

    reset(): print cummulative iterations and total runtime, then reset the
    interation count to 0 and start time to current time'''

    def __init__(self):
        self._iteration_count = 0
        self.__start_time = time.time()
        print('Calculator Initialized\n')

    def __del__(self):
        print('Calculator Destructed after',
              self._iteration_count, 'Iterations')
        print('Total Runtime:', int(time.time() - self.__start_time), 's\n')

    def reset(self):
        '''This function prints the cummulative iterations and total runtime,
        then resets the interation count to 0 and start time to current
        time.'''

        print('Calculator Reset after',
              self._iteration_count, 'Iterations')
        print('Total Runtime:', int(time.time() - self.__start_time), 's\n')
        self._iteration_count = 0
        self.__start_time = time.time()


class ReflectivityCalculator(Calculator):
    '''This derived class of calculator contains the functions used to
    calculate the 1D array of reflectivity from a multi-layered model.

    flatten_params(instr_broad, model): This function takes the instrumental
    broadening factor and the multilayered model as arguments. It returns a 1D
    list containing all the parameters, begging with instr_broad, to be
    optimized by the curve_fit function.

    refl_dist(self, qz, instr_broad, *params): This function takes the qz from
    experiment after cleaning (1D array), the instrumental broadening factor,
    and the model parameters as arguements. It calculates and returns the log10
    reflectivity assuming that the roughness of interfaces can be represented
    by exponential decay of the fresnell reflectivity coefficients. This
    is designed so it can be passed to curve_fit function.

    refl_cont(self, qz, instr_broad, *params): The arguements and return value
    of this function has exactly the same datatype. The only difference is that
    instead of assuming exponential decay of fresnell reflectivity
    coefficients, it use RCWA-like algorithm to calculate the reflectivity.

    optimize(self, calculator, instr_broad, data, model): This function takes
    a calculatro, either refl_dist or refl_cont as an arguement, together with
    the instrumental broadening factor, the instance of data and model. It
    returns the ptimal values for the parameters so that the sum of the squared
    residuals of f(xdata, *popt) - ydata is minimized, and the estimated
    covariance (2D array with diagonals provide the variance of the parameter
    estimate).'''
    # inline comments referred to the size of ndarrays

    @staticmethod
    # @jit(nopython=True)
    def _calc_wave(qz, sldr):
        '''This function takes a 1D array of qz and a 1D array of real SLD
        (sldr). It returns a 2D wavevector matrix (kz_i)'''

        kz_0 = qz / 2  # n_theta

        wave_length = 1.5406  # unit: A
        delta = wave_length ** 2 / 2 / np.pi * sldr  # n_layers
        theta_c = np.sqrt(2 * delta)  # n_layers
        kc = 2 * np.pi * np.sin(theta_c) / wave_length  # n_layers

        kz_sq = kz_0.reshape(-1, 1) ** 2 - kc ** 2  # n_theta, n_layers
        for i in range(kz_sq.shape[-1]):
            kz_sq[kz_sq[:, i] <= 0, i] = np.nan
        kz = np.sqrt(kz_sq)  # n_theta, n_layers
        # kz contatins nan at incident angle larger than the critical angle

        return kz

    @staticmethod
    # @jit(nopython=True)
    def _calc_refl_coef_ideal(kz):
        '''This function takes a 2D wavevector matrix (kz). It returns a 1D
        array of reflectivity coefficients for ideally sharp interfaces
        (refl_coef_ideal).'''

        refl_coef_ideal = ((kz[:, :-1] - kz[:, 1:]) /
                           (kz[:, :-1] + kz[:, 1:])).reshape(kz.shape[0], -1)
        # r_fres refers to the fresnell reflectivity coefficients at interfaces
        # r_fres has a dimension of n_theta, n_layers-1

        for i in range(refl_coef_ideal.shape[-1]):
            refl_coef_ideal[np.isnan(refl_coef_ideal[:, i]), i] = 1
        # It's written in this way instead of indexing with 2D boolean array
        # to be compatible with git. An element in r_fres indicates total
        # external reflection thus should be corrected to 1.
        return refl_coef_ideal

    @staticmethod
    # @jit(nopython=True)
    def _calc_refl_coef_diff(refl_coef_ideal, kz, msr):
        '''This function takes a 1D array of the reflectivity coefficients for
        ideally sharp interfaces (refl_coef_ideal), a 2D wavevector matrix
        (kz), and a 1D array of mean square roughness (msr). It returns a 1D
        array of reflectivity coefficients for diffused interfaces
        (refl_coef_diff).'''

        refl_coef_diff = (refl_coef_ideal *
                          np.exp(-2 * kz[:, :-1] * kz[:, 1:] * msr[1:] ** 2)
                          ).reshape(kz.shape[0], -1)

        for i in range(refl_coef_diff.shape[-1]):
            refl_coef_diff[np.isnan(refl_coef_diff[:, i]), i] = 1
        return refl_coef_diff

    @staticmethod
    # @jit(nopython=True)
    def _calc_refl(thick, refl_coef, kz):
        '''This function takes a 1D array of layer thickness, a 1D array of
        reflectivity coefficients (either ideal or diffused), a 2D wavevector
        matrix (kz_i). It returns a 1D array of reflectivity
        (refl).'''

        refl_coef_final = refl_coef[:, -1].astype(np.complex128)  # n_theta
        # Assign the fresnell reflectivity coefficient of the bottom layer as
        # the beginning point of the recursion.

        for i in range(kz.shape[-1] - 2):  # loops for n_layers - 2 times
            # Bottom-up interate through all interfaces beginning at the second
            # bottom layer because the real reflectivity coefficient at  first
            # bottom layer is the same as its fresnell reflectiviy coefficient.

            dividend = (refl_coef[:, -(2 + i)] +
                        refl_coef_final * np.exp(2j * thick[-(2 + i)]
                                                 * kz[:, -(2 + i)]))  # n_theta
            divisor = ((1 + refl_coef[:, -(2 + i)] *
                        refl_coef_final * np.exp(2j * thick[-(2 + i)]
                                                 * kz[:, -(2 + i)])))

            bool_isnan = np.isnan(dividend)  # n_theta
            dividend[bool_isnan] = 1
            divisor[bool_isnan] = 1

            refl_coef_final = dividend / divisor  # n_theta
            # The fresnel reflectivity coefficient matraix doesn't contain nan.
            # If the dividend or divisor contain nan, it must come from the kz
            # matrix, thus represent the total reflection. The index of nan in
            # dividend and divisor matrix is always the same.

        refl = (refl_coef_final * refl_coef_final.conj()).real

        return refl

    @staticmethod
    def flatten_params(instr_broad, model):
        '''This function prepares a flattened list of parameters, including
        instr_broad and thickness, msr, sldr, of a model. This flattened list
        is later used as the input for curve_fit. The data format after
        flattening is instr_broad, thick[1], msr[1], sldr[1], thick[2], msr[2],
        sldr[2], ..., thick[n-1], msr[n-1], sldr[n-1], msr[n], sldr[n].
        Parameters of top layer and the thickness of the bottom layer are not
        part of the optimization thus are exclded from this array.'''

        flattened_params = np.hstack((instr_broad,
                                      np.dstack((model.thick[1:-1],
                                                 model.msr[1:-1],
                                                 model.sldr[1:-1])).flatten(),
                                      model.msr[-1],
                                      model.sldr[-1]))
        return flattened_params

    def refl_dist(self, qz, instr_broad, *params):
        '''This function is used for optimization thus has to be compatible
        with curve_fit. params of the top layer and thickness of the bottom
        layer has to be excluded form the input arguements.

        This function takes a flattened 1D array, which used to be a n*3 2D
        array containing the thickness (thick), mean square roughness (msr),
        and real SLD (SLDr) of each layer. It returns a 1D array of
        reflectivity (refl). The impact of roughness on reflectivity is
        represented by an exponential decay of the reflectivity coefficients.
        The flattening of input array is for compatability with fitting'''

        self._iteration_count = self._iteration_count + 1

        params = np.array(params).flatten()
        params = np.hstack((0, 0, 0,
                            params[:-2],
                            0, params[-2], params[-1])).reshape(-1, 3)

        thick = params[:, 0]
        msr = params[:, 1]
        sldr = params[:, 2]

        kz = self._calc_wave(qz, sldr)
        refl_coef_ideal = self._calc_refl_coef_ideal(kz)
        refl_coef_diff = self._calc_refl_coef_diff(refl_coef_ideal, kz, msr)
        refl = self._calc_refl(thick, refl_coef_diff, kz)
        refl = gaussian_filter1d(refl, instr_broad)
        log10_refl = np.log10(refl)
        return log10_refl

    def refl_cont(self, qz, instr_broad, *params):
        '''This function is used for optimization thus has to be compatible
        with curve_fit. params of the top layer and thickness of the bottom
        layer has to be excluded form the input arguements.

        This function takes a flattened 1D array, which used to be a n*3 2D
        array containing the thickness (thick), mean square roughness (msr),
        and real SLD (SLDr) of each layer. It returns a 1D array of
        reflectivity (refl). The impact of roughness on reflectivity is
        represented by an exponential decay of the reflectivity coefficients.
        The flattening of input array is for compatability with fitting'''

        self._iteration_count = self._iteration_count + 1

        params = np.array(params).flatten()

        middle_params = np.hstack((params[:-2].reshape(-1, 3),
                                   np.zeros(params[:-2].reshape(-1, 3).shape[0]).reshape(-1, 1)))
        params_for_model = np.hstack((0, 0, 0, 0,
                                      middle_params.flatten(),
                                      0, params[-2], params[-1], 0))

        model = Model(True, 10000, params_for_model)
        new_model = model.discretize()
        new_params = np.dstack((new_model.thick, new_model.sldr)).flatten()
        new_params = np.array(new_params).reshape(-1, 2)

        thick = new_params[:, 0]
        sldr = new_params[:, 1]
        kz = self._calc_wave(qz, sldr)
        refl_coef_ideal = self._calc_refl_coef_ideal(kz)
        refl = self._calc_refl(thick, refl_coef_ideal, kz)
        refl = gaussian_filter1d(refl, instr_broad)
        log10_refl = np.log10(refl)
        return log10_refl

    def optimize(self, calculator, instr_broad, data, model):
        '''This function optimizes the model with a certain calculator (either
        dist or cont), so it will better fit the data. It returns popt and
        pcov after optimization.'''
        qz = data.qz
        log10_refl = np.log10(data.rfl)
        params = self.flatten_params(instr_broad, model)
        return curve_fit(calculator, qz, log10_refl, p0=params,
                         bounds=(0.8 * params, 1.2 * params))
