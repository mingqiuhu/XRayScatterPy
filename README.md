XRayScatterPy is a Python package for processing and analyzing x-ray scattering, grazing incidence x-ray scattering (GISAXS), and x-ray reflectivity data. It can process neutron scattering data in transmission and grazing incidence geometry. The package provides functionalities for reading raw image data, performing calibrations, plotting scattering data in 2D and 1D, calculating beam divergence, extracting specular reflectivity, fitting data with models, and assisting in thin film grating metrology. We intend to integrate bornagain package into this package in future developments. 

Modules:

calibration.py: This module contains functions to calculate the real and reciprocal space coordinates for each detector pixel.

data_processing.py: This module provides functions to run 1D average. 

data_plotting.py: This module contains functions for plotting the scattering data in 2D and 1D.

reflectivity.py: This module provides functionalities to extract specular reflectivity from a set of hundreds of GISAXS measurements at different incidence angles.

grating.py: This module provides functions for assisting in thin film grating metrology by predicting the position of the peaks of a grating.

utils.py: This module contains utility functions for general tasks like file I/O, image manipulation, and other miscellaneous tasks.

setup.py: This file contains the package setup information for installation.

README.md: This file provides an overview of the package, installation instructions, and usage examples.

Installation:
You can install the XRayScatterPy package using pip:
Firstly navigate to the root directory, the directoy containing the setup file.
Then run the following command in terminal.
pip install -e .

Contributing:
We welcome contributions to the XRayScatterPy package. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

Documentation:
We are now prepareing a paper aimed at Journal of Polymer Science, and the detailed documentation is also under preparation. 

Dependencies:
The XRayScatterPy package relies on the following external libraries:

NumPy: For efficient numerical operations and data manipulation
SciPy: For scientific computing and optimization
Matplotlib: For data visualization and plotting
tifffile: To read and write tiff images
xmltodict: To parse the headers of an experiment. 
requests: To access the NIST website to get the scattering length density of different materials.
These dependencies will be automatically installed when you install the XRayScatterPy package using pip.

Support:
If you encounter any issues or have questions related to the XRayScatterPy package, please open an issue on the GitHub repository or reach out to the maintainers at mingqiuhu@mail.pse.umass.edu or xgan@mail.pse.umass.edu

License:
The XRayScatterPy package is released under the MIT License.

By providing a comprehensive suite of functionalities for x-ray scattering, grazing incidence x-ray scattering, and x-ray reflectivity data analysis, the XRayScatterPy package aims to streamline your research and analysis process. We hope you find this package useful, and we look forward to your feedback and contributions.