This repository contains a python pipeline which takes row-stacked spectra and creates strictly flux-conserved data cubes by using 
column-normalized matrix multiplication which maps fibers to pixels. This package was created specifically for the LVM but could be used for 
any IFU (in theory). 

Each python file works as follows. 

1. io_readers.py: creates a FiberSet, where a FiberSet is a Class that contains np.ndarrays of flux, ivar (inverse-variance), wavelength,
   and x_arcsec, y_arcsec positions in the IFU plane, from the fibers. FiberSet also contains an extra meta field which is a python Dict
   containing any extra data you want to pass into the FiberSet. This can be generated from either data analysis fluxes or
   reduced row-stacked-spectra.
3. kernels.py: Build a sparse matrix A with shape (P, K), where P = (ny*nx) no. of pixels, and K = no. of fibers.
   Column K holds all weights for each fiber k, the weights are built from a radial kernel (typically gaussian, but tophat is an option)
   and the sum of weights = 1


   To be continued...

The resulting cubes can be in units of flux/pixel, or surface brightness, flux/arcsec^2, surface brightness makes more sense, and cubes are 
automatically stored in sb. 
