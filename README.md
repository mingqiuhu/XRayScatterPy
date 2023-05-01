XRayScatterPy is a Python package for processing and analyzing x-ray scattering, grazing incidence x-ray scattering (GISAXS), and x-ray reflectivity data. It can process neutron scattering data in transmission and grazing incidence geometry. The package provides functionalities for reading raw image data, performing calibrations, plotting scattering data in 2D and 1D, calculating beam divergence, extracting specular reflectivity, fitting data with models, and assisting in thin film grating metrology. We intend to integrate bornagain package into this package in future developments. 

Modules:
calibration.py: This module contains functions to calculate the real and reciprocal space coordinates for each detector pixel.
data_processing.py: This module provides functions to calculate 1D average, and normalize the scattering intensity.
data_plotting.py: This module contains functions for plotting the scattering data in 2D and 1D.
reflectivity.py: This module provides functionalities to extract specular reflectivity from a set of hundreds of GISAXS measurements at different incidence angles.
grating.py: This module provides functions for assisting in thin film grating metrology by predicting the position of the peaks of a grating.
utils.py: This module contains utility functions for general tasks like file I/O, and type checking.


Installation:
You can download and install the XRayScatterPy package using pip:
In a terminal, firstly navigate to the root directory, the directoy containing the setup file.
Then run the following command in the terminal, including the "." at the end of the line.
pip install -e .

Contributing:
We welcome contributions to the XRayScatterPy package. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

Documentation:
This is a pre-release of the codes. We are now prepareing a paper aimed at Journal of Polymer Science, and the detailed documentation is also under preparation. We anticipate a detailed documentation to be available in a few months. Please feel free to try out the examples. We recommende running the examples in a python interactive kernel, for example the one in vscode.

Dependencies:
The XRayScatterPy package relies on the following external libraries:

numpy: For efficient numerical operations and data manipulation
scipy: For scientific computing and optimization
matplotlib: For data visualization and plotting
tifffile: To read and write tiff images
xmltodict: To parse the headers of an experiment. 
requests: To access the NIST website to get the scattering length density of different materials.
These dependencies will be automatically installed when you install the XRayScatterPy package using pip install -e .

Support:
If you encounter any issues or have questions related to the XRayScatterPy package, please open an issue on the GitHub repository or reach out to the maintainers at mingqiuhu@mail.pse.umass.edu or xgan@mail.pse.umass.edu

License:
The XRayScatterPy package is released under the MIT License.

By providing a comprehensive suite of functionalities for x-ray scattering, grazing incidence x-ray scattering, and x-ray reflectivity data analysis, the XRayScatterPy package aims to streamline your research and analysis process. We hope you find this package useful, and we look forward to your feedback and contributions.
