# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:36:13 2022

@author: pkuhu
"""
import numpy as np
import matplotlib.pyplot as plt
import reflectivity_pltlib


class Data1d:
    '''Original and cleaned experiment data from a txt file, where reflectivity
    (2nd column) is a function of qz (1st column). The cleaning process removes
    all nan and 0 elements from the reflectivity profile and the corresponding
    qz.

    qz_org: 1D array, original qz from experiment data.

    rfl_org: 1D array, original reflectivity from experiment data.

    qz: 1D array, cleaned qz.

    rfl: 1D array, cleaned reflectivity.

    plt_org(): plot the original reflectivity as a function of qz.

    plt_clean(): plot the cleaned reflectivity as a function of cleaned qz.'''

    def __init__(self, file):

        def read(file):
            data_org = np.genfromtxt(file)
            return data_org[:, 0], data_org[:, 1]

        self.qz_org, self.rfl_org = read(file)

        def clean():
            data_delete = (np.isnan(self.qz_org) |
                           np.isnan(self.rfl_org) | (self.rfl_org == 0))
            # index of the qz and reflectivity to be deleted

            return self.qz_org[~data_delete], self.rfl_org[~data_delete]

        self.qz, self.rfl = clean()

    def plt_org(self):
        '''plot original reflectivity as a function of qz'''

        reflectivity_pltlib.plt_set_params_before()
        plt.plot(self.qz_org, self.rfl_org, label='Original Data')
        reflectivity_pltlib.plt_set_params_after_refl()

    def plt_clean(self):
        '''plot cleaned reflectity as a function of qz'''

        reflectivity_pltlib.plt_set_params_before()
        plt.plot(self.qz, self.rfl, label='Cleaned Data')
        reflectivity_pltlib.plt_set_params_after_refl()
