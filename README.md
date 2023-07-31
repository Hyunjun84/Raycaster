# Raycaster
Raycaster with Six direction Box-spline and FCC Voronoi Spline.


Six direction Box-Spline : https://dx.doi.org/10.2139/ssrn.4146126

FCC Voronoi Spline : https://doi.org/10.1007/s11075-023-01562-5

# Library Dependency

- numpy
- pyopencl
- pyopengl

install with pip : 

	python3 -m pip install numpy pyopencl pyopengl

# Running
	./sh raycaster.sh

# Usage

- Mouse action
	- drag - rotation
	- drag with shift - translation on xy-plane
	- drag with control - scaling
	- wheel - change FOV
- Keyboard action
	 - 'k' - change kernel
	 - 'v' - change volume data
	 - 's' - change shader(Blnn-Phong, Min/max curvature)
	 - '-' - decrease isovalue
	 - '=' - increase isovalue
	 - 'q' - on/off quasi-interpolation
	 - 'x' - take screenshot
	 - ESC - exit





