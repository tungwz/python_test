import numpy as np
from scipy.io import FortranFile


def bin2netcdf(file_name, variable):

    # all variable of binary file

    f = FortranFile(file_name, 'r')  # read the binary file

    a[[[]]] = f.read_reals()

    return a


print(bin2netcdf(r'C:\Users\adminX\Desktop\model_landtypes.bin', 'fraction_patch'))
