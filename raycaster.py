"""
raycaster.py

# Copyright (c) 2023, Hyunjun Kim
# All rights reserved.

"""

import numpy as np
import pyopencl as cl
import logging
from lattice import Lattice


class Raycaster :
    def __init__(self, ctx, devices, queue, title, kernel_src, resolution, sampling_lattice, qi_coefficients) :
        self.ctx = ctx
        self.queue = queue
        self.title = title
        self.lattice = sampling_lattice
        self.qi_coeff = qi_coefficients
        self.Log = logging.getLogger("Raycaster")

        self.prg = cl.Program(self.ctx, kernel_src)
        self.prg.build(options=["",],devices=[devices[0],], cache_dir=None)
        self.setResolution(resolution)
        
    def setResolution(self, siz) :
        mf = cl.mem_flags
        self.dim_ray = siz
        self.rays = cl.Buffer(self.ctx, mf.READ_WRITE, np.prod(siz)*8*4)

    def dumpRayBuffer(self) :
        nr_rays = np.prod(self.dim_ray)
        buf = np.zeros([nr_rays,8], dtype=np.float32)
        cl.enqueue_copy(self.queue, src=self.rays, dest=buf)
        self.queue.finish()
        np.set_printoptions(precision=2)
        length = [np.linalg.norm(buf[i,:4] - buf[i,4:]) for i in range(nr_rays)]
        print(max(length))
        nr_valid = [i for i in length if not i == 0]
        print(len(nr_valid))
        print(sum(nr_valid))
   
    def raycast(self, iso, buf_pos, volume_data, MVP, data_ratio) :
        buf_vol, dim = volume_data
        return self.prg.raycast(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=buf_pos,
            arg1=buf_vol,
            arg2=np.float32(MVP),
            arg3=np.float32(data_ratio),
            arg4=np.float32(dim),
            arg5=np.float32(iso),
        )

    def evalGradient(self, buf_grad, buf_pos, volume_data, data_ratio) :
        buf_vol, dim = volume_data
        return self.prg.evalGradient(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=buf_grad, 
            arg1=buf_pos,
            arg2=buf_vol,
            arg3=np.float32(dim),
            arg4=np.float32(data_ratio))

    def evalHessian(self, buf_H1, buf_H2, buf_pos, volume_data, data_ratio) :
        buf_vol, dim = volume_data
        return self.prg.evalHessian(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=buf_H1, 
            arg1=buf_H2,
            arg2=buf_pos,
            arg3=buf_vol,
            arg4=np.float32(data_ratio),
            arg5=np.float32(dim))
