"""
raycaster.py

# Copyright (c) 2023, Hyunjun Kim
# All rights reserved.

"""

import logging
import numpy as np
import pyopencl as cl
from enum import Enum

from lattice import Lattice

class VolumeData :
    def __init__(self, ctx, devices, queue) :
        self.ctx = ctx
        self.queue = queue
        self.Log = logging.getLogger("Raycaster")

        self.h_data = None
        self.d_data = None
        self.d_data_QI = None

        with open('./kernel/volume_data.cl', 'r') as fp : src = fp.read()
        self.prg = cl.Program(self.ctx, src)
        self.prg.build(options=["",],devices=[devices[0],], cache_dir=None)

    def getVolumeData(self, withQI=False) :
        if withQI : return (self.d_data_QI, self.dim)
        else : return (self.d_data, self.dim)

    def uploadVolumeData(self, title, FILE_PATH, dim, dtype, orientation, scale, lattice, model=np.eye(4,dtype=np.float32), isovalue=0) :
        self.title = title
        self.h_data = np.fromfile(FILE_PATH, dtype=dtype).astype(np.float32)
        self.dim = dim
        self.volume_res = dim
        self.orientation = orientation
        self.scale = scale
        self.data_ratio = [i*j / max(self.dim) for i,j in zip(self.dim, self.scale)]
        self.isovalue = isovalue
        self.model = model
        self.lattice = lattice

        if not self.d_data == None : self.d_data.release()
        if not self.d_data_QI == None : self.d_data_QI.release()

        match(lattice) :
            case Lattice.CC :
                mf = cl.mem_flags
                fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
                self.d_data = cl.Image(context=self.ctx, flags=mf.READ_ONLY|mf.COPY_HOST_PTR, format=fmt, shape=self.dim[:3], hostbuf=self.h_data)
                self.d_data_QI = cl.Image(context=self.ctx, flags=mf.READ_WRITE, format=fmt, shape=self.dim[:3])

            case Lattice.FCC :
                # note that axis order of numpy array.(z,y,x)
                self.h_data = np.reshape(self.h_data, [dim[2], dim[1], dim[0], 1])
                self.volume_res = [self.dim[0]//2, self.dim[1]//2, self.dim[2]//2, 4]

                # fcc offset : 000, 011, 101, 110
                # note that axis order of numpy array.(z,y,x) 
                offset = [[0,0,0], [1,1,0], [1,0,1], [0,1,1]]
                h_fcc = [np.array(self.h_data[o[0]::2, o[1]::2, o[2]::2]) for o in offset]
                h_fcc = np.concatenate(h_fcc, axis=3, dtype=np.float32)
                
                mf = cl.mem_flags
                fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
                self.d_data = cl.Image(context=self.ctx, flags=mf.READ_ONLY|mf.COPY_HOST_PTR, format=fmt, shape=self.volume_res[:3], hostbuf=h_fcc)
                self.d_data_QI = cl.Image(context=self.ctx, flags=mf.READ_WRITE, format=fmt, shape=self.volume_res[:3])

                self.dim = dim

        self.Log.info("Volume data infomation :")
        self.Log.info("\tresolution : ({0}, {1}, {2})".format(*dim))
        self.Log.info("\trange : {0:.4f} - {1:.4f}".format(np.min(self.h_data), np.max(self.h_data)))
        self.Log.info("\tscale : ({0}, {1}, {2})".format(*scale))
        self.Log.info("\tratio : ({0}, {1}, {2})".format(*self.data_ratio))
        self.Log.info("\tnormalized domain : ({0}, {1}, {2})".format(*self.data_ratio))



    def uploadVolumeDataWithConfig(self, config) :
        self.uploadVolumeData(config.name,
                              config['path'],
                              eval(config['resolution']),
                              config['type'],
                              eval(config['orientation']),
                              eval(config['scale']),
                              eval(config['sampling lattice']),
                              eval(config['model']),
                              eval(config['isovalue']))

    def applyQuasiInterpolator(self, coeff) :
        sz_global = np.array(self.volume_res[:3]).astype(np.int32)
        sz_local = np.array([4,4,4]).astype(np.int32)

        sz_global = tuple(sz_global-1 + sz_local-(sz_global-1)%sz_local)
        sz_local = tuple(sz_local)

        if self.lattice == Lattice.CC : 
            evt = self.prg.applyQuasiInterpolator_CC(queue=self.queue,
                                               global_size=sz_global,
                                               local_size=sz_local,
                                               arg0=self.d_data_QI,
                                               arg1=self.d_data,
                                               arg2=np.float32(coeff),
                                               arg3=np.int32(self.dim))
            
        if self.lattice == Lattice.FCC : 
            evt = self.prg.applyQuasiInterpolator_FCC(queue=self.queue,
                                               global_size=sz_global,
                                               local_size=sz_local,
                                               arg0=self.d_data_QI,
                                               arg1=self.d_data,
                                               arg2=np.float32(coeff),
                                               arg3=np.int32(self.dim))
            
        return evt
