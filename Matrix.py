import numpy as np

def translate(t):
    return np.array((
        (1, 0, 0, t[0]), 
        (0, 1, 0, t[1]),
        (0, 0, 1, t[2]), 
        (0, 0, 0, 1),
        ), dtype=np.float32)

def scale(s) :
    return np.array((
        (s[0], 0, 0, 0),
        (0, s[1], 0, 0),
        (0, 0, s[2], 0),
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
        ( (W(0,0)+ca),    (W(0,1)-ws[2]), (W(0,2)+ws[1]), 0 ),
        ( (W(0,1)+ws[2]), (W(1,1)+ca),    (W(1,2)-ws[0]), 0 ),
        ( (W(0,2)-ws[1]), (W(1,2)+ws[0]), (W(2,2)+ca),    0 ),
        ( 0,              0,              0,              1 )
        ), dtype=np.float32)

def lookAt(eye, center, up) :
    f = np.array(eye)-np.array(center)
    f = f/np.linalg.norm(f)
    l = np.cross(up, f)
    l = l/np.linalg.norm(l)

    u = np.cross(f, l)

    return np.array([[l[0], l[1], l[2], -np.dot(l,eye)],
                     [u[0], u[1], u[2], -np.dot(u,eye)],
                     [f[0], f[1], f[2], -np.dot(f,eye)],
                     [   0,    0,    0,           1.0]],
                    dtype=np.float32)

def inverse(M) :
    return np.linalg.inv(M)

def perspective(fov, aspect, near, far) :
    p00 = 1.0/np.tan(fov/2.0)
    p11 = p00/aspect
    p22 = (far+near)/(near-far)
    p23 = (2*near*far)/(near-far)
    return np.array([[p00, 0.0,  0.0,  0.0],
                     [0.0, p11,  0.0,  0.0],
                     [0.0, 0.0,  p22,  p23],
                     [0.0, 0.0, -1.0,  0.0]],
                    dtype=np.float32)


def ortho(l, r, b, t, n ,f) :
    return np.array([[2/(r-l),       0,       0, (l+r)/(l-r)],
                     [      0, 2/(t-b),       0, (b+t)/(b-t)],
                     [      0,       0, 2/(n-f), (n+f)/(n-f)],
                     [      0,       0,       0,          1]],
                    dtype=np.float32)
