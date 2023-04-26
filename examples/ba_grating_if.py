#!/usr/bin/env python3
"""
Long boxes at 1D lattice, ba.Offspec simulation
"""
import bornagain as ba
from bornagain import ba_plot as bp, deg, nm


def get_sample():
    """
    Returns a sample with a grating on a substrate,
    modelled by infinitely long boxes forming a 1D lattice.
    """

    # Define materials
    material_Particle = ba.RefractiveMaterial("Particle", 0.0006, 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6e-06, 2e-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0, 0)

    # Define form factors
    ff = ba.Box(1000 * nm, 20 * nm, 10 * nm)

    # Define particles
    particle = ba.Particle(material_Particle, ff)
    particle_rotation = ba.RotationZ(90 * deg)
    particle.rotate(particle_rotation)

    # Define interference functions
    iff = ba.Interference1DLattice(100 * nm, 0)
    iff_pdf = ba.Profile1DCauchy(1e6 * nm)
    iff.setDecayFunction(iff_pdf)

    # Define particle layouts
    layout = ba.ParticleLayout()
    layout.addParticle(particle)
    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(0.01)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_1.addLayout(layout)
    layer_2 = ba.Layer(material_Substrate)

    # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)

    return sample


def get_simulation(sample):
    """
    Returns an off-specular simulation with beam and detector defined.
    """
    n = bp.simargs['n']
    scan = ba.AlphaScan(n, 0.1 * deg, 10 * deg)
    scan.setIntensity(1e9)
    scan.setWavelength(0.1 * nm)
    detector = ba.OffspecDetector(
        n, -1 * deg, +1 * deg, n, 0.1 * deg, 10 * deg)
    return ba.OffspecSimulation(scan, sample, detector)


if __name__ == '__main__':
    bp.parse_args(sim_n=200, intensity_min=1)
    sample = get_sample()
    simulation = get_simulation(sample)
    result = simulation.simulate()
    bp.plot_simulation_result(result)
