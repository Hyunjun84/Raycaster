import logging 
import numpy as np
import pyopencl as cl


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

    def uploadVolumeData(self, FILE, dim, typ) :
        self.h_data = np.fromfile(FILE, dtype=typ).astype(np.float32)
        self.dim = dim

        self.Log.info("Volume data infomation :")
        self.Log.info("\tdimension : ({0}, {1}, {2})".format(*dim))
        self.Log.info("\trange : {0:.4f} - {1:.4f}".format(np.min(self.h_data), np.max(self.h_data)))
        
        if not self.d_data == None : self.d_data.release()
        if not self.d_data_QI == None : self.d_data_QI.release()

        mf = cl.mem_flags
        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)

        self.d_data = cl.Image(context=self.ctx, flags=mf.READ_ONLY|mf.COPY_HOST_PTR, format=fmt, shape=self.dim[:3], hostbuf=self.h_data)
        self.d_data_QI = cl.Image(context=self.ctx, flags=mf.READ_WRITE, format=fmt, shape=self.dim[:3])

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
    