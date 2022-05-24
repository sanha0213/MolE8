===============================================================

MolE8_v2.py Manual

Version 2.0

Copyright (c) 2019 Sanha Lee, Kristaps Ermanis and Jonathan M. Goodman
University of Cambridge

This document is adapted from 'MolE8: Finding DFT Potential Energy Surface Minima Values from Force-Field Optimised Organic Molecules with New Machine Learning Representations' publication


* Note for GitHub download *

The code requires ML_Models directory to run. This can be downloaded from Cambridge Apollo repository: https://doi.org/10.17863/CAM.78009


===============================================================

CONTENTS
1) Requirements and Setup
2) Usage
3) Example Usage

===============================================================


1) REQUIREMENTS AND SETUP

This script is currently set up to run on Python version 3.7. The following Python modules must be installed before the script can be run:
- keras
- sys
- numpy
- pickle
- openbabel
- argparse
- logging
- time

The Python script is designed to calculate the energies or free energies of the molecules in the .xyz file.


2) CORRECT USAGE

Download the directories Feat_Dicts and ML_Models and place them at the same location where MolE8_v1.py code will be running. If the directories are not placed in the same location as the MolE8_v1.py code, the 'database_loc' and 'dictionary_loc' in the 'Settings' section can be edited to direct the python code to the correct directory.

The .xyz file and the instructions must be inputed in the following order:

python MolE8_v1.py Filname.xyz Model_Option Energy_Option

Filename.xyz - name of the xyz file to calculate the properties, the .xyz extension must be typed

Model_Option - trained machine learning model to be used to calculate the properties. Type NN for neural network model, KRR for kernel ridge regression model or MLR for multivariable linear regression model.

Training_Option - select between ML models trained with distorted molecules replicates (d) or trained without distorted molecule replicates (wd)

Energy_Option - type E to predict the SCF energy at B3LYP/6-31G(2df,p) or type G to predict the free energy at B3LYP/6-31G(2df,p)

The algorithm will create a log file and print out the predicted properties in the order they appear in the .xyz file. The log file will have the name syarting MolE8_v1_...

The algorithm will also generate:
- MolE8_ReadableXMatrix_ : features vector generated for each molecule in the xyz file in txt format, which can be opened and visualised easily in text editor
- MolE8_ReadableXFeatures_ : this file shows what each elements in the features vector represents
- MolE8_X_ : features vector generated for each molecule in the xyz file in pickle format


3) EXAMPLE USAGE

The following files are provided for this section:

- TestMolecules_Energies_xyz.xyz : Example two molecules to perform SCF energy ML calculation 

- TestMolecules_Energies_DFT_values.txt : SCF energy of the example two molecules at B3LYP/6-31G(2df,p) level calculated from Gaussian 16 for comparison

- TestMolecules_FreeEnergies_DFT_values.txt : Free energy of the example two molecules at B3LYP/6-31G(2df,p) level calculated from Gaussian 16 for comparison

- X_matrices: directory containing expected features matrices for the two example molecules provided. Readable files are also provided so the user can compare the feature vector to the bonds, angles, dihedrals etc. These files are also generated when the MolE8.py code is run


We will firstly test the code on the example xyz file by predicting the SCF energy using the kernel ridge regression trained without the distorted molecule replicates:

python MolE8_v1.py TestMolecules_Energies_xyz.xyz KRR wd E

Compare the outputted energy by checking the TestMolecules_Energies_DFT_values file - the energy values should be very similar.

We can also compare the features X matrix by checking the MolE8_v1_ReadableXMatrix_TestMolecules_Energies_xyz_KRR_E_wd.txt file generated after running the code to TestMolecules_Energies_Xs.txt. These two matrices should be identical.

We can also test the code on the example xyz file by predicting the SCF energy using the neural network model:

python MolE8_v1.py TestMolecules_Energies_xyz.xyz NN wd E

Again check whether the predicted energies are similar to the Gaussian SCF energy.

Finally we can test the code on the example xyz file by predicting the free energy using the kernel ridge regression and neural network models:
python MolE8_v1.py TestMolecules_Energies_xyz.xyz KRR wd G
python MolE8_v1.py TestMolecules_Energies_xyz.xyz NN wd G