XRayScatterPy is a Python package for processing and analyzing x-ray scattering, grazing incidence x-ray scattering (GISAXS), and x-ray reflectivity data. It can process neutron scattering data in transmission and grazing incidence geometry. The package provides functionalities for reading raw image data, performing calibrations, plotting scattering data in 2D and 1D, calculating beam divergence, extracting specular reflectivity, fitting data with models, and assisting in thin film grating metrology. We intend to integrate bornagain package into this package in future developments. 
  
# Installation:
Python 3 is required for running the XRayScatterPy package. After downloading and unzipping the XRayScatterPy package, you can install it in a virtual environment using `pip` and the built-in `venv` module with terminal tools.  
For MacOS and Linux users, the system built-in terminal tool is recommanded; for Windows users, `git bash` is recommanded.  
In a terminal, firstly use `cd` command to navigate to the work directory, where you want to run your scripts and keep your data in.  
Then run the following commands in the terminal to create a virtual environment:
```
python -m venv <name_of_venv>
```
you can replace `<name_of_venv>` with an arbitrary name for the virtual environment, like `xrsp`.  
Next, activate the virtual environment with the following command,  
for MacOS and Linux, use:
```
source <name_of_venv>/bin/activate
```
for Windows, run in git bash:
```
source <name_of_venv>/Scripts/activate
```
Then install the downloaded package:
```
pip install <directory_to_package>
```
Replace `<directory_to_downloaded_package>` with the directory to the downloaded and unzipped package, for example: 
/Users/username/Documents/XRayScatterPy  
You can find out the currect directory with `pwd` command.

# Using the package:
For running the package, firstly navigate to the working directory and activate the virtual environment.
Then, write the python script with any text edit tool or IDE. you can find examples of scripts in `XRayScatterPy/examples/`.  
For example, `gisaxs.py` is an example script of processing raw GISAXS data; `transmission_scattering.py` is the example for generating 2D scattering patterns for transmission X-ray scattering measurements;  
then run the script with:
```
python script.py
```

# Modules:  
`calibration.py`: This module contains functions to calculate the real and reciprocal space coordinates for each detector pixel.  
`data_processing.py`: This module provides functions to calculate 1D average, and normalize the scattering intensity.  
`data_plotting.py`: This module contains functions for plotting the scattering data in 2D and 1D.  
`reflectivity.py`: This module provides functionalities to extract specular reflectivity from a set of hundreds of GISAXS measurements at different incidence angles.  
`grating.py`: This module provides functions for assisting in thin film grating metrology by predicting the position of the peaks of a grating.  
`utils.py`: This module contains utility functions for general tasks like file I/O, and type checking.  

# Contributing:  
We welcome contributions to the `XRayScatterPy` package. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.  
  
# Documentation:  
This is a pre-release of the codes, the GUI and detailed documentation are under preparation. Please feel free to try out the examples.

# Dependencies:  
The XRayScatterPy package relies on the following external libraries:  
  
`numpy`: For efficient numerical operations and data manipulation.  
`scipy`: For scientific computing and optimization.  
`matplotlib`: For data visualization and plotting.  
`tifffile`: To read and write tiff images.  
`xmltodict`: To parse the headers of an experiment.  
`requests`: To access the NIST website to get the scattering length density of different materials.  
These dependencies will be automatically installed when you install the `XRayScatterPy` package.  
  
# Support:  
If you encounter any issues or have questions related to the XRayScatterPy package, please open an issue on the GitHub repository or reach out to the maintainers at mingqiuhu@mail.pse.umass.edu or xgan@mail.pse.umass.edu.  
  
# License:  
The XRayScatterPy package is released under the MIT License.  

By providing a comprehensive suite of functionalities for x-ray scattering, grazing incidence x-ray scattering, and x-ray reflectivity data analysis, the XRayScatterPy package aims to streamline your research and analysis process. We hope you find this package useful, and we look forward to your feedback and contributions.

Please cite the following paper if you use this package for data processing and plotting in publications:

Hu, M.; Gan, X.; Chen, Z.; Seong, H.-G.; Emrick, T.; Russell, T. P. Quantitative X‐ray Scattering and Reflectivity Measurements of Polymer Thin Films with 2D Detectors. J. Polym. Sci. 2024, 62(16), 3642-3662. 
[https://doi.org/10.1002/pol.20230530](https://doi.org/10.1002/pol.20230530)
