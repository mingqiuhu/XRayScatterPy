#!/usr/bin/env python3
"""
Simulation of grating using very long boxes and 1D lattice.
Monte-carlo integration is used to get rid of
large-particle form factor oscillations.
"""
import bornagain as ba
from bornagain import angstrom, ba_plot as bp, deg, micrometer, nm
from matplotlib import pyplot as plt


def get_sample(lattice_rotation_angle=0*deg):
    """
    Returns a sample with a grating on a substrate.
    lattice_rotation_angle = 0 - beam parallel to grating lines
    lattice_rotation_angle = 90*deg - beam perpendicular to grating lines
    """
    # defining materials
    m_vacuum = ba.RefractiveMaterial("Vacuum", 0, 0)
    m_si = ba.RefractiveMaterial("Si", 5.7816e-6, 1.0229e-7)

    box_length, box_width, box_height = 50*micrometer, 70*nm, 50*nm
    lattice_length = 150*nm

    # collection of particles
    interference = ba.Interference1DLattice(
        lattice_length, 90*deg - lattice_rotation_angle)

    pdf = ba.Profile1DGauss(450)
    interference.setDecayFunction(pdf)

    box_ff = ba.LongBoxLorentz(box_length, box_width, box_height)
    box = ba.Particle(m_si, box_ff)
    box.rotate(ba.RotationZ(lattice_rotation_angle))

    particle_layout = ba.ParticleLayout()
    particle_layout.addParticle(box)
    particle_layout.setInterference(interference)

    # assembling the sample
    vacuum_layer = ba.Layer(m_vacuum)
    vacuum_layer.addLayout(particle_layout)
    substrate_layer = ba.Layer(m_si)

    sigma, hurst, corrLength = 5*nm, 0.5, 10*nm
    roughness = ba.LayerRoughness(sigma, hurst, corrLength)

    sample = ba.MultiLayer()
    sample.addLayer(vacuum_layer)
    sample.addLayerWithTopRoughness(substrate_layer, roughness)
    return sample


def get_simulation(sample):
    beam = ba.Beam(1e8, 1.34*angstrom, 0.4*deg)
    n = bp.simargs['n']
    det = ba.SphericalDetector(n, -0.5*deg, 0.5*deg, n, 0, 0.6*deg)
    simulation = ba.ScatteringSimulation(beam, sample, det)
    simulation.options().setMonteCarloIntegration(True, 100)
    return simulation


if __name__ == '__main__':
    bp.parse_args(sim_n=401)
    sample = get_sample()
    simulation = get_simulation(sample)
    if not "__no_terminal__" in globals():
        simulation.setTerminalProgressMonitor()
    result = simulation.simulate()

    field = result.datafield()
    bp.plot_histogram(field, with_cb=False)

    peaks = ba.FindPeaks(field, 2, "nomarkov", 0.001)
    xpeaks = [peak[0] for peak in peaks]
    ypeaks = [peak[1] for peak in peaks]
    print(peaks)

    plt.plot(xpeaks,
             ypeaks,
             marker='x',
             linestyle='none',
             color='white',
             markersize=10)

    bp.show_or_export()