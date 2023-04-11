# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:42:13 2022

@author: pkuhu
"""
import numpy as np
from reflectivity_input import Data1d
from reflectivity_model import Model
import reflectivity_pltlib
from reflectivity_calculate import ReflectivityCalculator


def main():
    data = Data1d(r'D:\OneDrive - University of Massachusetts\Umass Amherst'
                  r'\Research\Data\XRR_Processed\Processed'
                  r'\03122022 XRR Si - SiO2(100nm) - W(80-100nm) from Shiva.txt')

    instr_broad = 1.3
    md1 = Model(True, 10000,
                0,   0,     0, 0,
                653.8, 9, 122e-6, 0,
                0,   4.7, 20.017e-6, 0)

    calculator = ReflectivityCalculator()

    param_for_calc = calculator.flatten_params(instr_broad, md1)
    r_dist0 = calculator.refl_dist(data.qz, param_for_calc[0], param_for_calc[1:])
    #r_cont0 = calculator.refl_cont(data.qz, param_for_calc[0], param_for_calc[1:])

    #calculator.reset()

    popt_dist, pcov_dist = calculator.optimize(calculator.refl_dist, instr_broad, data, md1)
    
    #calculator.reset()

    #popt_cont, pcov_cont = calculator.optimize(calculator.refl_cont, instr_broad, data, md1)

    #calculator.reset()

    r_dist = calculator.refl_dist(data.qz, popt_dist[0], popt_dist[1:])
    #r_cont = calculator.refl_dist(data.qz, popt_cont[0], popt_cont[1:])
    
    del calculator

    print('popt_dist', popt_dist)
    print('pcov_dist', np.sqrt(np.diag(pcov_dist)))
    #print('popt_cont', popt_cont)
    #print('pcov_cont', np.sqrt(np.diag(pcov_cont)))

    reflectivity_pltlib.plt_refl_model(data.qz, data.rfl, 10**r_dist0, 'Experiment', 'r_dist0')
    #pltlib.plt_refl_model(data.qz, data.rfl, 10**r_cont0, 'Experiment', 'r_cont0')
    reflectivity_pltlib.plt_refl_model(data.qz, data.rfl, 10**r_dist, 'Experiment', 'Model')
    #pltlib.plt_refl_model(data.qz, data.rfl, 10**r_cont, 'Experiment', 'r_cont')

    data.plt_org()
    data.plt_clean()

    md1.plt_ideal()
    md1.plt_broadened()
    md1.plt_all()


if __name__ == "__main__":
    main()
