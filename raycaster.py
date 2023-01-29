import numpy as np
import pyopencl as cl
import logging

class Raycaster :
    def __init__(self, ctx, devices, queue, splines) :
        self.ctx = ctx
        self.queue = queue
        self.prgs = []
        self.Log = logging.getLogger("Raycaster")

        with open('./kernel/raycaster.cl', 'r') as fp : src = fp.read()
        for title, sp in splines.items() :
            with open(sp, 'r') as fp : 
                kernel_src = src+fp.read()
                prg = cl.Program(self.ctx, kernel_src)
                prg.build(options=["",],devices=[devices[0],], cache_dir=None)
                self.prgs.append((title, prg))

        self.current_prg_idx = -1
        self.nextKernel()
        self.setNumberofRays([8,8])
        
    def nextKernel(self) :
        self.current_prg_idx = (self.current_prg_idx+1)%len(self.prgs)
        self.prg = self.prgs[self.current_prg_idx][1]
        self.Log.info("Current kernel : {0}".format(self.prgs[self.current_prg_idx][0]))

    def setNumberofRays(self, siz) :
        mf = cl.mem_flags
        self.dim_ray = siz
        self.rays = cl.Buffer(self.ctx, mf.READ_WRITE, np.prod(siz)*8*4)

    def uploadVolumeData(self, FILE, dim, typ) :
        self.data = np.fromfile(FILE, dtype=typ).astype(np.float32)
        self.dim = dim

        self.Log.info("Volume data infomation :")
        self.Log.info("\tdimension : ({0}, {1}, {2})".format(*dim))
        self.Log.info("\trange : {0:.4f} - {1:.4f}".format(np.min(self.data), np.max(self.data)))
        
        mf = cl.mem_flags
        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        print(self.data)
        self.d_volume = cl.Image(context=self.ctx, flags=mf.READ_ONLY|mf.COPY_HOST_PTR, format=fmt, shape=self.dim[:3], hostbuf=self.data)

    def ray_dump(self) :
        buf = np.zeros([512*512,8], dtype=np.float32)
        cl.enqueue_copy(self.queue, src=self.rays, dest=buf)
        self.queue.finish()
        np.set_printoptions(precision=2)
        for i in range(10) :
                print(buf[i,:])

    def debug_dump(self) :
        buf = np.zeros([512*512,8], dtype=np.int32)
        cl.enqueue_copy(self.queue, src=self.temp, dest=buf)
        self.queue.finish()
        print(np.min(buf, axis=0), np.max(buf, axis=0))

    def genRay(self, MVP, fov) :
        return self.prg.genRay(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=self.rays,
            arg1=MVP,
            arg2=fov
        )

    def raycast(self, iso, buf_pos) :
        return self.prg.raycast(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=buf_pos,
            arg1=self.d_volume,
            arg2=self.rays,
            arg3=np.float32([1,1,1,1]),
            arg4=np.int32(self.dim),
            arg5=np.float32(iso),
        )

    def evalGradient(self, buf_pos, buf_grad) :
        return self.prg.evalGradient(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=buf_grad, 
            arg1=self.d_volume,
            arg2=buf_pos,
            arg3=np.int32(self.dim))


if __name__ == "__main__":
    import logging
    import numpy as np
    import pyopencl as cl
    from pyopencl.tools import get_gl_sharing_context_properties

    Log = logging.getLogger("Raycaster_mod_test")
    Log.setLevel(logging.DEBUG)
    hFileLog = logging.FileHandler("./output/raycaster_mod.log")
    hStreamLog = logging.StreamHandler()
    formatter = logging.Formatter(fmt='[%(levelname)s][%(asctime)s.%(msecs)03d][%(funcName)s():%(lineno)d] %(message)s', datefmt='%H:%M:%S')
    hFileLog.setFormatter(formatter)
    hStreamLog.setFormatter(formatter)
    Log.addHandler(hFileLog)
    Log.addHandler(hStreamLog)


    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx_properties = [(cl.context_properties.PLATFORM, platforms[0])]
    ctx = cl.Context(dev_type=cl.device_type.GPU, properties=ctx_properties)
    queue = cl.CommandQueue(context=ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


    raycaster = Raycaster(ctx, devices, queue)


    mf = cl.mem_flags
    d_ray = cl.Buffer(ctx, mf.READ_WRITE, 512*512*4*8)
    d_fbuf = cl.Buffer(ctx, mf.READ_WRITE, 512*512*4)

    MVP = np.eye(4).astype(np.float32)
    th = np.pi/3
    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(th), np.sin(th), 0],
            [0, -np.sin(th), np.cos(th), 0],
            [0, 0, 0, 1]
        ]
        ).astype(np.float32)

    Ry = np.array(
        [
            [np.cos(th), 0, np.sin(th), 0],
            [0, 1, 0, 0],
            [-np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1]
        ]
        ).astype(np.float32)

    MVP = np.dot(MVP,np.dot(Rx, Ry))


    raycaster.setNumberofRays(512*512)
    raycaster.genRay([512,512], MVP)
    raycaster.shading(d_fbuf, [512,512])


    h_ray = np.zeros([512*512,8], dtype=np.float32)
    h_fbuf = np.zeros([512*512], dtype=np.float32)

    cl.enqueue_copy(queue=queue,src=d_ray, dest=h_ray)
    cl.enqueue_copy(queue=queue,src=d_fbuf, dest=h_fbuf)

    queue.finish()

    for i in range(8) :
        print("[{0:.4f} {1:.4f} {2:.4f} {3:.4f}] <-> [{4:.4f} {5:.4f} {6:.4f} {7:.4f}]".format(*list(h_ray[i,:])))
    