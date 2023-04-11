XRayScatterPy is a Python package for processing and analyzing x-ray scattering, grazing incidence x-ray scattering (GISAXS), and x-ray reflectivity data. The package provides functionalities for reading raw image data, performing calibrations, plotting scattering data in 2D and 1D, calculating beam divergence, extracting specular reflectivity, fitting data with models, and assisting in thin film grating metrology. It also supports data fitting using the BornAgain package.

Modules:

calibration.py: This module contains functions to read calibration data from silver behenate and glassy carbon standard reference material for q-calibration and absolute intensity calibration.

data_processing.py: This module provides functions for processing raw image data, performing 1D integration in q or azimuthal angle, and calculating beam divergence based on geometry and experimental data.

data_plotting.py: This module contains functions for plotting the scattering data in 2D (qy, qz or q parallel, qz) and 1D, as well as transverse scattering, longitudinal scattering, and off-specular reflectivity.

reflectivity.py: This module provides functionalities to extract specular reflectivity from a set of hundreds of GISAXS measurements at different incidence angles.

model_fitting.py: This module contains functions for fitting the data with models, including support for BornAgain fitting.

grating.py: This module provides functions for assisting in thin film grating metrology by predicting the position of the peaks of a grating.

utils.py: This module contains utility functions for general tasks like file I/O, image manipulation, and other miscellaneous tasks.

setup.py: This file contains the package setup information for installation.

README.md: This file provides an overview of the package, installation instructions, and usage examples.

Installation:
You can install the XRayScatterPy package using pip:
pip install xray-scatter-py


Contributing:
We welcome contributions to the XRayScatterPy package. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

Documentation:
Detailed documentation for the XRayScatterPy package can be found at https://xray-scatter-py.readthedocs.io. The documentation includes a comprehensive guide for each module and function, as well as examples to help you get started quickly.

Dependencies:
The XRayScatterPy package relies on the following external libraries:

NumPy: For efficient numerical operations and data manipulation
SciPy: For scientific computing and optimization
Matplotlib: For data visualization and plotting
scikit-image: For image processing and manipulation
pandas: For data handling and analysis
BornAgain: For fitting the data with models (optional)
These dependencies will be automatically installed when you install the XRayScatterPy package using pip.

Support:
If you encounter any issues or have questions related to the XRayScatterPy package, please open an issue on the GitHub repository or reach out to the maintainers through the contact information provided in the README file.

License:
The XRayScatterPy package is released under the MIT License. For more information, please refer to the LICENSE file in the repository.

By providing a comprehensive suite of functionalities for x-ray scattering, grazing incidence x-ray scattering, and x-ray reflectivity data analysis, the XRayScatterPy package aims to streamline your research and analysis process. We hope you find this package useful, and we look forward to your feedback and contributions.
