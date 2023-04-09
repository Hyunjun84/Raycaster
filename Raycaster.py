import numpy as np
import pyopencl as cl
import logging
import glm

class Raycaster :
    def __init__(self, ctx, devices, queue, kernel_src, resolution) :
        self.ctx = ctx
        self.queue = queue
        self.prgs = []
        self.Log = logging.getLogger("Raycaster")

        self.prg = cl.Program(self.ctx, kernel_src)
        self.prg.build(options=["",],devices=[devices[0],], cache_dir=None)
        
        self.setResolution(resolution)
        
    def setResolution(self, siz) :
        mf = cl.mem_flags
        self.dim_ray = siz
        self.rays = cl.Buffer(self.ctx, mf.READ_WRITE, np.prod(siz)*8*4)

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

    def genRay(self, MVP) :
        return self.prg.genRay(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=self.rays,
            #arg1=glm.transpose(MVP)
            arg1=MVP
        )

    def raycast(self, iso, buf_pos, volume_data) :
        buf_vol, dim = volume_data
        return self.prg.raycast(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=buf_pos,
            arg1=buf_vol,
            arg2=self.rays,
            arg3=np.float32([1,1,1,1]),
            arg4=np.float32(dim),
            arg5=np.float32(iso),
        )

    def evalGradient(self, buf_grad, buf_pos, volume_data) :
        buf_vol, dim = volume_data
        return self.prg.evalGradient(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=buf_grad, 
            arg1=buf_pos,
            arg2=buf_vol,
            arg3=np.float32(dim))

    def evalHessian(self, buf_H1, buf_H2, buf_pos, volume_data) :
        buf_vol, dim = volume_data
        return self.prg.evalHessian(
            queue=self.queue, 
            global_size=self.dim_ray, 
            local_size=None, 
            arg0=buf_H1, 
            arg1=buf_H2,
            arg2=buf_pos,
            arg3=buf_vol,
            arg4=np.float32(dim))

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
    