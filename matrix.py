import numpy as np

def Identity() :
	return np.array((
		(1, 0 ,0, 0),
		(0, 1 ,0, 0),
		(0, 0 ,1, 0),
		(0, 0 ,0, 1),
		), dtype=np.float32)

def translate(x, y, z):
	return np.array((
		(1, 0, 0, x), 
		(0, 1, 0, y),
		(0, 0, 1, z), 
		(0, 0, 0, 1),
		), dtype=np.float32)

def scale(x, y, z) :
	return np.array((
		(x, 0, 0, 0),
		(0, y, 0, 0),
		(0, 0, z, 0),
		(0, 0, 0, 1),
		), dtype=np.float32)

def rotate(axis, angle) :
	w = axis/np.linalg.norm(axis)
	ca = np.cos(angle)
	sa = np.sin(angle)
	oca = 1-np.cos(angle)
	ws = w*sa
	W = (lambda x,y: w[x]*w[y]*oca)
	return np.array((
		( (W(0,0)+ca),    (W(0,1)+ws[2]), (W(0,2)-ws[1]), 0 ),
		( (W(0,1)-ws[2]), (W(1,1)+ca),    (W(1,2)+ws[0]), 0 ),
		( (W(0,2)+ws[1]), (W(1,2)-ws[0]), (W(2,2)+ca),    0 ),
		( 0,              0,              0,              1 )
		), dtype=np.float32)


def inverse(m) :
	return np.linalg.inv(m)


def transpose(m) :
	return np.transpose(m, axes=(1,0))
