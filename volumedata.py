import logging
import numpy as np
import pyopencl as cl
from enum import Enum

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

    def uploadVolumeData(self, title, FILE_PATH, dim, dtype, orientation, scale, model=np.eye(4,dtype=np.float32), isovalue=0) :
        self.title = title
        self.h_data = np.fromfile(FILE_PATH, dtype=dtype).astype(np.float32)
        self.dim = dim
        self.orientation = orientation
        self.scale = scale
        self.data_ratio = [i*j / max(self.dim) for i,j in zip(self.dim, self.scale)]
        self.isovalue = isovalue
        self.model = model

        self.Log.info("Volume data infomation :")
        self.Log.info("\tresolution : ({0}, {1}, {2})".format(*dim))
        self.Log.info("\trange : {0:.4f} - {1:.4f}".format(np.min(self.h_data), np.max(self.h_data)))
        self.Log.info("\tscale : ({0}, {1}, {2})".format(*scale))
        self.Log.info("\tnormalized domain : ({0}, {1}, {2})".format(*self.data_ratio))

        if not self.d_data == None : self.d_data.release()
        if not self.d_data_QI == None : self.d_data_QI.release()

        mf = cl.mem_flags
        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)

        self.d_data = cl.Image(context=self.ctx, flags=mf.READ_ONLY|mf.COPY_HOST_PTR, format=fmt, shape=self.dim[:3], hostbuf=self.h_data)
        self.d_data_QI = cl.Image(context=self.ctx, flags=mf.READ_WRITE, format=fmt, shape=self.dim[:3])

    def uploadVolumeDataWithConfig(self, config) :
        title = config.name

        self.uploadVolumeData(config.name,
                              config['path'],
                              eval(config['resolution']),
                              config['type'],
                              eval(config['orientation']),
                              eval(config['scale']),
                              eval(config['model']),
                              eval(config['isovalue']))

    def applyQuasiInterpolator(self, coeff) :
        sz_global = np.array(self.dim[:3]).astype(np.int32)
        sz_local = np.array([4,4,4]).astype(np.int32)

        sz_global = tuple(sz_global-1 + sz_local-(sz_global-1)%sz_local)
        sz_local = tuple(sz_local)

        self.prg.applyQuasiInterpolator(queue=self.queue,
                                        global_size=sz_global,
                                        local_size=sz_local,
                                        arg0=self.d_data_QI,
                                        arg1=self.d_data,
                                        arg2=np.float32(coeff),
                                        arg3=np.int32(self.dim))

