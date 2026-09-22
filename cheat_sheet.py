# Emission-line map(s) from a DAP file
python -m ifuCube.build dap-maps \
  --dap dap-00004163.dap.fits.gz --fibermap fibermap.yaml \
  --out map.fits --lines Halpha "[NII]6583" \
  --pix 2.0 --fwhm 3.2 --frad 18.5 --rsigma 3 --kernel gauss

# Cube from an SFrame
python -m ifuCube.build cube \
  --sframe sframe-00003299.fits --fibermap fibermap.yaml \
  --out cube.fits --pix 2.0 --fwhm 3.2 --frad 18.5 --rsigma 3 --kernel gauss

# Ratio map from an existing multi-line map FITS
python -m ifuCube.build ratio \
  --in-map map.fits --num "[NII]6583" --den Halpha --out ratio.fits

# MOSAICING

# Maps (2D)
python -m ifuCube.mosaic \
  --inputs map1.fits map2.fits map3.fits \
  --hdu FLUX --weights-hdu IVAR \
  --output mosaic_map.fits \
  --mode map --auto-wcs --combine mean --reproject exact --flux-per-pixel

# Cubes (3D) — same idea, --mode cube
python -m ifuCube.mosaic \
  --inputs cube1.fits cube2.fits cube3.fits \
  --hdu FLUX --weights-hdu IVAR \
  --output mosaic_cube.fits \
  --mode cube --auto-wcs --combine mean --reproject exact

# SDSS downloads

from sdss_access import RsyncAccess
rsync = RsyncAccess(release="sdsswork")
rsync.lookup_keys("lvm_dap")   # what keys a filetype needs
rsync.full("lvm_dap", tileid=..., mjd=..., dapver=..., rspid=..., snlevel=..., daptype=..., expnum=...)  # resolved path

rsync.remote()
rsync.add("lvm_dap", tileid=..., mjd=..., dapver=..., 
          rspid=..., snlevel=..., daptype=..., expnum=...
          )  # resolved path

e.g. rsync.add('lvm_dap', mjd='60212', snlevel='sn20', rspid='rsp108', dapver='1.2.1', 
               daptype='dap', expnum='00005511', tileid='11111')
rsync.set_stream()
rsync.commit()

# Large healpix map configuration

PIXEL_SIZE_ARCSEC = 15.0     # tangent-plane pixels (≤ ~31" so no fibers are dropped)
FWHM_ARCSEC = 35.0           # ~one LVM fiber diameter
FIBER_RADIUS_ARCSEC = 18.5
R_SIGMA_MULT = 3.0
KERNEL_MODE = "gauss"
NSIDE = 4096                 # 51.5" HEALPix pixels
N_POINTINGS_TO_USE = 1000
MAPCAT_DIR = "mapcats_nside4096"
DRY_RUN = True               # check the download plan first

# og parameters

# --- Map-building parameters (kernels.py / geometry.py) ---
PIXEL_SIZE_ARCSEC = 2.0
FWHM_ARCSEC = 3.2
FIBER_RADIUS_ARCSEC = 18.5
R_SIGMA_MULT = 3.0
KERNEL_MODE = "gauss"
FIBERMAP_YAML = "lvm_fiducial_fibermap.yaml"
LINES = ["Halpha"]                 # single line -> HEALPix column will be "VALUE", not "VALUE0"

# --- HEALPix ---
#NSIDE = 131072                     # n = 131072 matches PIXEL_SIZE_ARCSEC=2.0 (see suggest_nside)
NSIDE = 4096                        # n = 4096 combine with psa=15" and kern of 40

# --- Batch size / safety ---
N_POINTINGS_TO_USE = 50            # slice of matched pointings to actually process
MAPCAT_DIR = "mapcats"              # all map_*.fits / hp_*.fits land here, not the cwd
HP_DIR = "healpix"                 # directory for HEALPix files
DRY_RUN = True                     # keep True until the printed download plan looks right