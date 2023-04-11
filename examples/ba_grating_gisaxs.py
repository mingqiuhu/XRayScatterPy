#!/usr/bin/env python3
"""
Simulation of grating using very long boxes and 1D lattice.
Monte-carlo integration is used to get rid of
large-particle form factor oscillations.

Maybe try spherical detector to reduce error
"""
import bornagain as ba
from bornagain import angstrom, ba_plot as bp, deg, micrometer, nm
from matplotlib import pyplot as plt
from xray_scatter_py import nist
import numpy as np
import os
from xray_scatter_py import data_plotting, utils, calibration, gratings
from scipy.ndimage import convolve


def get_sample(lattice_rotation_angle, full_pitch, pitch_width, pitch_height):
    """
    Returns a sample with a grating on a substrate.
    lattice_rotation_angle = 0 - beam parallel to grating lines
    lattice_rotation_angle = 90*deg - beam perpendicular to grating lines
    """
    # Define materials
    sld_si = nist.get_scattering_parsed("Si", "2.33")
    material_vacuum = ba.MaterialBySLD("Vacuum", 0, 0)
    material_si = ba.MaterialBySLD("Si", sld_si['x_real'], sld_si['x_imag'])

    # Define the form factor of a pitch of the grating
    pitch_length = 1e4*micrometer
    pitch_ff = ba.LongBoxLorentz(pitch_length, pitch_width, pitch_height)

    # Define a pitch of the grating with rotation
    pitch = ba.Particle(material_si, pitch_ff)
    pitch.rotate(ba.RotationZ(lattice_rotation_angle))

    # Define interference function of the grating
    interference = ba.Interference1DLattice(
        full_pitch, 90*deg - lattice_rotation_angle)
    interference_pdf = ba.Profile1DGauss(10*full_pitch) # half width of the decay of the 1D layout
    interference.setDecayFunction(interference_pdf)

    # Define layouts
    pitch_layout = ba.ParticleLayout()
    pitch_layout.addParticle(pitch)
    pitch_layout.setInterference(interference)

    # Define top roughness of the substrate layer
    sigma = 5*nm # rms of the roughness
    hurst = 0.5 # describe how jagged the interface is, 0 gives more spikes and 1 gives more smoothness
    corrLength = 2*nm  # lateral correlation length of the roughness
    roughness = ba.LayerRoughness(sigma, hurst, corrLength)

    # Define layers
    vacuum_layer = ba.Layer(material_vacuum)
    vacuum_layer.addLayout(pitch_layout)
    substrate_layer = ba.Layer(material_si)

    # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(vacuum_layer)
    sample.addLayerWithTopRoughness(substrate_layer, roughness)
    return sample


def get_simulation_ganesha(sample):
    # 1406507
    # hg1 0.399987
    # vg1 0.399987
    # hg3 0.200000
    # vg3 0.200000
    # 2833606
    # hg1 0.399987
    # vg1 0.399987
    # hg3 0.300000
    # vg3 0.300000
    # 16751498
    # hg1 0.899987
    # vg1 0.899987
    # hg3 0.400000
    # vg3 0.400000
    # 38985039
    # hg1 0.899987
    # vg1 0.899987
    # hg3 0.900000
    # vg3 0.900000
    beam = ba.Beam(1e8, 1.542*angstrom, 0.2*deg)
    detPos = 1472.587500  # distance from sample center to detector in mm
    detWid = 0.172  # detector width in mm
    detector = ba.RectangularDetector(619, 619*detWid, 487, 487*detWid)
    detector.setPerpendicularToDirectBeam(detPos, 335.3*detWid, (487-424.48)*detWid)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    simulation.options().setMonteCarloIntegration(True, 100)
    # simulation.options().setIncludeSpecular(True)
    return simulation


def create_kernel(radius):
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - radius) ** 2 + (j - radius) ** 2)
            if distance <= radius:
                kernel[i, j] = 1
    kernel /= np.sum(kernel)
    return kernel

def smooth_colormap(input_array, radius):
    kernel = create_kernel(radius)
    output_array = convolve(input_array, kernel)
    return output_array

if __name__ == '__main__':
    bp.parse_args()
    sample = get_sample(-5.5*deg, 139*nm, 70*nm, 50*nm)
    simulation = get_simulation_ganesha(sample)
    result = simulation.simulate()
    simulated_image_array = np.transpose(result.array())


    # 78465 - 78665 PHI = -5.5
    # 78670 - 78870 PHI = 4.8
    # 78875 - 79075 PHI = -0.04
    # 79080 - 79280 PHI = 90

    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tiff_files_grating')

    DETX0 = 100.4
    INDEX_LIST = [0]

    START_INDEX = 78645
    PHI = -5.5

    END_INDEX = START_INDEX


    params_dict_list, image_array = utils.read_multiimage(DATA_PATH, START_INDEX, END_INDEX)
    theta_array, azimuth_array = calibration.calculate_angle(DETX0, params_dict_list, image_array)
    qx_array, qy_array, qz_array = calibration.calculate_q(DETX0, params_dict_list, image_array)
    omega = calibration.calculate_omega(DETX0, params_dict_list, theta_array)
    image_array_rel = calibration.calibrate_rel_intensity(params_dict_list, image_array, omega)

    radius = 2
    simulated_image_array = smooth_colormap(simulated_image_array, radius)

    cutoff_intensity = np.max(simulated_image_array) * 1e-7
    simulated_image_array[simulated_image_array < cutoff_intensity] = 0


    data_plotting.plot_2d_scattering(qy_array, qz_array, image_array_rel, index_list=INDEX_LIST)
    data_plotting.plot_2d_scattering(qy_array, qz_array, simulated_image_array[np.newaxis,:,:], index_list=INDEX_LIST)

    # bp.plot_simulation_result(result)