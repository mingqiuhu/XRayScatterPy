# tests/test_data_plotting.py

import os
import numpy as np
from xray_scatter_py import data_plotting, utils, calibration, data_processing
import matplotlib.pyplot as plt


thickness_dict = {
    'A_1_1_20s_exposure_2m':0.048,
    'A_1_20s_exposure_5_2m':0.05,
    'A_2_1_20s_exposure_2m':0.053,
    'A_2_20s_exposure_2_2m':0.05,
    'A_3_1_20s_exposure_2m':0.05,
    'A_3_20s_exposure_2m':0.056,
    'A_4_1_20s_exposure_2m':0.05,
    'A_4_20s_exposure_2m':0.056,
    'A_5_1_20s_exposure_2_2m':0.049,
    'A_5_20s_exposure_2m':0.054,
    'A_6_20s_exposure_2m':0.055,
    'B_1_1_20s_exposure_2m':0.055,
    'B_1_20s_exposure_2_2m':0.053,
    'B_2_1_20s_exposure_2m':0.059,
    'B_2_20s_exposure_3_2m':0.055,
    'B_3_1_20s_exposure_2_2m':0.063,
    'B_3_20s_exposure_2m':0.054,
    'B_4_20s_exposure_2m':0.052,
    'B_6_20s_exposure_2m':0.054,
    'C_1_1_20s_exposure_2m':0.055,
    'C_1_20s_exposure_2_2m':0.05,
    'C_2_20s_exposure_2m':0.057,
    'C_3_20s_exposure_2m':0.052,
    'D_1_20s_exposure_2m':0.05,
    'D_3_20s_exposure_2_2m':0.053,
    'D_3_20s_exposure_2m':0.053,
    'D_4_20s_exposure_2_2m':0.05,
    'D_4_20s_exposure_2m':0.05,
    'ST_20s_exposure_2_2m':0.1055,
    'ST_60s_exposure_2m':0.1055
}

Q_MIN = 0.00827568
Q_MAX = 0.24740200
Q_NUM = 59

def calculate_synchrotron_abs(FILE_NAME, thickness, Q_MIN, Q_MAX, Q_NUM):
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'synchrotron')
    image = utils.read_synchrotron_image(DATA_PATH, FILE_NAME+'.edf_mod.tif')
    with open(os.path.join(DATA_PATH, FILE_NAME+'.txt'), 'r') as file:
        I0 = float(file.readline())
        DiodeNorm = float(file.readline())
    factor = 53954218.44159385
    image = 10**image
    image[np.isnan(image)] = -1
    image[1150:, 650:700] = -1
    image_array = image[np.newaxis, :, :]
    params_dict = {}
    params_dict['beamcenter_actual'] = '[1178.31 682.88]'
    params_dict['pixelsize'] = '[0.172 0.172]'
    params_dict['detx'] = '3524.18'
    params_dict['det_exposure_time'] = '1'
    params_dict['wavelength'] = '1.23984'
    params_dict['sample_transfact'] = str(DiodeNorm/I0)
    params_dict['sample_thickness'] = str(thickness)
    params_dict['saxsconf_Izero'] = str(I0)
    params_dict_list = [params_dict]

    theta_array, azimuth_array = calibration.calculate_angle(0, params_dict_list, image_array)
    qx_array, qy_array, qz_array = calibration.calculate_q(0, params_dict_list, image_array)
    q_array = np.sqrt(qx_array**2 + qy_array**2 + qz_array**2)
    omega = calibration.calculate_omega(0, params_dict_list, theta_array)
    image_array_rel = calibration.calibrate_rel_intensity(params_dict_list, image_array, omega)
    image_array_abs = calibration.calibrate_abs_intensity(params_dict_list, image_array_rel)

    q_1d = np.linspace(Q_MIN, Q_MAX, Q_NUM)
    i_1d = data_processing.calculate_1d(Q_MIN, Q_MAX, Q_NUM, q_array, image_array_abs, omega)
    np.savetxt(FILE_NAME+'.txt', np.stack((q_1d, i_1d[0, :]/factor), axis=-1))
    
    data_plotting.plot_2d_scattering(-qz_array, qy_array, image_array_abs/factor, video=False)

    srm = np.loadtxt(os.path.join(DATA_PATH, 'srm.csv'), delimiter=',')
    data_plotting.plot_set()
    plt.figure()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(q_1d, i_1d[0, :]/factor, '.')
    plt.plot(srm[:,0], srm[:,1])
    plt.plot(srm[:,0], srm[:,1]+srm[:,2], '--', color=cycle[1])
    plt.plot(srm[:,0], srm[:,1]-srm[:,2], '--', color=cycle[1])
    plt.xlabel(r'$q\ \mathrm{(Å^{-1})}$', fontsize=22)
    plt.ylabel(r'$I\ \mathrm{(cm^{-1}sr^{-1})}$', fontsize=22)
    plt.xscale('log')
    plt.yscale('linear')
    plt.legend(['measured', 'reference', '95% confidence range'], fontsize=20)
    plt.show()

    

# for FILE_NAME, thickness in thickness_dict.items():
#     calculate_synchrotron_abs(FILE_NAME, thickness, Q_MIN, Q_MAX, Q_NUM)

def calculate_synchrotron_abs_cheap(FILE_NAME, thickness, Q_MIN, Q_MAX, Q_NUM):
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'synchrotron')
    image = utils.read_synchrotron_image(DATA_PATH, FILE_NAME+'.edf_mod.tif')
    with open(os.path.join(DATA_PATH, FILE_NAME+'.txt'), 'r') as file:
        I0 = float(file.readline())
        DiodeNorm = float(file.readline())
    factor = 53954218.44159385
    image = 10**image
    image[np.isnan(image)] = -1
    image[1150:, 650:700] = -1
    image_array = image[np.newaxis, :, :]
    params_dict = {}
    params_dict['beamcenter_actual'] = '[1178.31 682.88]'
    params_dict['pixelsize'] = '[0.172 0.172]'
    params_dict['detx'] = '3524.18'
    params_dict['det_exposure_time'] = '1'
    params_dict['wavelength'] = '1.23984'
    params_dict['sample_transfact'] = str(DiodeNorm/I0)
    params_dict['sample_thickness'] = str(thickness)
    params_dict['saxsconf_Izero'] = str(I0)
    params_dict_list = [params_dict]

    theta_array, azimuth_array = calibration.calculate_angle(0, params_dict_list, image_array)
    qx_array, qy_array, qz_array = calibration.calculate_q(0, params_dict_list, image_array)
    q_array = np.sqrt(qx_array**2 + qy_array**2 + qz_array**2)
    omega = calibration.calculate_omega(0, params_dict_list, theta_array)
    image_array_rel = calibration.calibrate_rel_intensity(params_dict_list, image_array, omega)
    image_array_abs = calibration.calibrate_abs_intensity(params_dict_list, image_array_rel)

    q_1d = np.linspace(Q_MIN, Q_MAX, Q_NUM)
    i_1d = data_processing.calculate_1d_cheap(Q_MIN, Q_MAX, Q_NUM, q_array, image_array_abs, omega)
    np.savetxt(FILE_NAME+'.txt', np.stack((q_1d, i_1d[0, :]/factor), axis=-1))


    data_plotting.plot_2d_scattering(-qz_array, qy_array, image_array_abs/factor, video=False)

    srm = np.loadtxt(os.path.join(DATA_PATH, 'srm.csv'), delimiter=',')
    data_plotting.plot_set()
    plt.figure()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(q_1d, i_1d[0, :]/factor, '.')
    plt.plot(srm[:,0], srm[:,1])
    plt.plot(srm[:,0], srm[:,1]+srm[:,2], '--', color=cycle[1])
    plt.plot(srm[:,0], srm[:,1]-srm[:,2], '--', color=cycle[1])
    plt.xlabel(r'$q\ \mathrm{(Å^{-1})}$', fontsize=22)
    plt.ylabel(r'$I\ \mathrm{(cm^{-1}sr^{-1})}$', fontsize=22)
    plt.xscale('log')
    plt.yscale('linear')
    plt.legend(['measured', 'reference', '95% confidence range'], fontsize=20)
    plt.show()

# calculate_synchrotron_abs('ST_60s_exposure_2m', 0.1055, Q_MIN, Q_MAX, 400)
calculate_synchrotron_abs_cheap('ST_60s_exposure_2m', 0.1055, Q_MIN, Q_MAX, 300)