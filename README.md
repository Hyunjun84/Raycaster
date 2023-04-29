# Raycaster
Raycaster with Six direction Box-spline and FCC Voronoi Spline.

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





