# ifuCube

A Python pipeline that turns row-stacked fibre spectra into strictly flux-conserving data cubes and emission-line maps. It was written for the SDSS-V Local Volume Mapper, but nothing about the method is specific to LVM and it should work for any fibre IFU.

## How it works

LVM observes with 1801 fibres of 35.3 arcsec diameter on a 37 arcsec hexagonal lattice. The fibres do not tile the sky — they cover about 83 per cent of it — and they are not on a rectangular grid, so turning fibre spectra into an image is a resampling problem.

Each fibre's flux is spread over nearby pixels with a radial kernel, and all those weights are assembled into one sparse matrix that maps fibres to pixels. The matrix is normalised column by column, so the weights belonging to any single fibre sum to exactly one. That is the whole design, and everything else follows from it: a fibre's flux is shared out among pixels and none of it is lost, so summing the pixels of a region gives back the total flux in that region.

This is a deposition, not an interpolation. The value in a pixel answers "how much flux was assigned here", not "what is the brightness at this point". Those are different questions, and the second one needs the other normalisation — dividing by the stored coverage map turns the deposition into a local weighted average. Both are available, because which one you want depends on the quantity. Flux adds up and must be deposited; velocity does not add up and must be averaged, which is why kinematic maps use a different operator built from the same matrix.

## Units and conventions

Cubes are stored as surface brightness, in flux per square arcsecond, so the values do not depend on the pixel size the cube happened to be built with. Maps are stored as flux per pixel, the conserving quantity, and the readers convert on the way back in. The headers record which convention a file uses, so nothing has to be guessed and no file is ever converted twice.

Maps and cubes come out in the standard sky orientation, North up and East left.

Two things are worth knowing before thresholding anything. The coverage map is a sum of kernel weights, not a fraction, and its typical value over the field is small — around 0.19 for 15 arcsec pixels — so cutting on a raw coverage value will mask the entire map. Use the normalised coverage instead, which is close to one across the well-sampled interior whatever the pixel size and kernel width. Separately, fibre apertures cover only part of the sky, so a surface brightness read off a map is a little below the true sky value by a constant factor. It cancels in every line ratio and in any comparison between LVM maps, and matters only in absolute units against another survey or a model.

## What is in the package

Reading and representing the data:

- **io_readers** — the FiberSet container that everything else speaks: fluxes, inverse variances, wavelengths, fibre positions, and a metadata dictionary for anything else you want to carry along.
- **io_lvm** — readers for LVM reduced spectra and DAP files. These work out which fibres are science fibres and which are healthy from the file itself, so a map does not depend on an external fibre map that may have changed since the exposure. Every fibre in the input is recorded as kept or rejected, and the headers note which column supplied the decision.
- **constants** — the instrument numbers, in one place so nothing else has to own them.

Building:

- **geometry** — the pixel grid, the inside-the-field mask, and the celestial WCS.
- **kernels** — the projection matrix itself, the forward projection, the error propagation, and the effective number of fibres per pixel.
- **projection** — the setup step both builders share, so it exists once rather than twice.
- **cube_build** — three-dimensional cubes.
- **mapcat_build** — emission-line maps, kinematic maps, and a per-fibre catalogue written alongside them.
- **fill** — estimates the flux of a rejected fibre from its neighbours, for when a gap-free picture is wanted. It refuses to guess inside a large contiguous gap rather than inventing structure there, and flags every fibre it imputed so they can be excluded later.

Afterwards:

- **ratio** — line-ratio maps with propagated errors.
- **regions** — pick fibres out by circle or polygon and coadd their spectra.
- **mosaic** — combine overlapping pointings onto a common grid.
- **healpix_build** — move maps and cubes onto a HEALPix grid, and merge and plot the results.
- **diagnostics** — a validation suite to run on anything the builders produce. It checks that no flux was lost, that the operator and its adjoint agree, and that every fibre's weights sum to one.
- **plotting** — quick-look figures for maps, cubes and fibre catalogues.

The two builders can also be run from the command line.

## Requirements

NumPy, SciPy and Astropy are needed. Reading LVM files additionally needs polars, PyYAML and the LVM DAP package, which is not on PyPI. Plotting needs matplotlib, the HEALPix step needs astropy-healpix, and mosaicking needs reproject. All of these are imported only where they are used, so a machine that just builds cubes and writes FITS does not need a plotting stack.

## Known issues

A footprint centroid is computed as a plain mean of longitudes in the HEALPix crossmatch, which is wrong for a field straddling RA = 0. The mosaic coadd reopens every input file once per wavelength plane, which is far more reads than necessary and worth fixing before running a large mosaic.

## Reference

Drory et al. 2024, AJ 168, 198, for the LVM instrument.
