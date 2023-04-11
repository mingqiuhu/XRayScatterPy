# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 02:34:17 2022

@author: pkuhu
"""
import matplotlib
import matplotlib.pyplot as plt


R_ticks = [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0]


def plt_set_params_before():
    '''This function sets the parameters for plotting before plt.plot().'''

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['figure.dpi'] = 600
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'legend.fontsize': 18})
    plt.figure()


def plt_set_params_after_refl():
    '''This function sets the parameters afer plotting reflectivity as a
    function of qz.'''

    plt.legend()
    plt.yscale('log')
    plt.ylim(10**-7, 10)
    plt.xlim(0.02, 0.4)
    plt.yticks(R_ticks)
    plt.xlabel('q'+'$_{z}$'+' (A'+'$^{-1}$'+')')
    plt.ylabel('R')
    plt.show()

def plt_set_params_after_off_specular():
    '''This function sets the parameters afer plotting reflectivity as a
    function of qz.'''

    plt.legend()
    plt.yscale('log')
    # plt.ylim(10**-7, 10)
    plt.xlim(0, 3)
    # plt.yticks(R_ticks)
    plt.xlabel(r'$\alpha_f\ (^{\circ})$')
    plt.ylabel('I (a.u.)')
    plt.show()
    
def plt_set_params_after_rock():
    '''This function sets the parameters afer plotting reflectivity as a
    function of qz.'''

    plt.legend()
    plt.yscale('log')
    # plt.ylim(10**-7, 10)
    plt.xlim(-1.5, 1.5)
    # plt.yticks(R_ticks)
    plt.xlabel(r'$\alpha_i-\phi/2\ (^{\circ})$')
    plt.ylabel('I (a.u.)')
    plt.show()
    
def plt_set_params_after_R():
    '''This function sets the parameters afer plotting reflectivity as a
    function of qz.'''

    plt.legend()
    plt.yscale('log')
    plt.ylim(10**-8, 1.1)
    plt.xlim(0.02, 0.3)
    # plt.yticks(R_ticks)
    plt.xlabel(r'$q_z\ (A^{-1})$')
    plt.ylabel('R')
    plt.show()


def plt_set_params_after_sld():
    '''This function sets the parameters after plotting SLD as a function of
    Depth.'''

    plt.legend()
    plt.xlabel('Depth (A)')
    plt.ylabel('SLD (10' + '$^{-6}$' + 'A' + '$^{-2}$' + ')')
    plt.show()


def plt_refl_model(qz, refl, refl_diff, label1, label2):
    '''This function plots two reflectivity profiles with labels as a function
    of qz.'''

    plt_set_params_before()
    plt.plot(qz, refl, label=label1)
    plt.plot(qz, refl_diff, label=label2)
    plt_set_params_after_refl()
