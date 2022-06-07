import numpy as np
import pyopencl as cl

class Raycaster :
    def __init__(self, ctx, devices, queue) :
        self.ctx = ctx
        self.queue = queue
        with open('./kernel/raycaster.cl', 'r') as fp : src = fp.read()
        self.prg = cl.Program(self.ctx, src)
        self.prg.build(options=["-g",],devices=[devices[0],], cache_dir=None)
        
        
    def Shading(self, frame_buffer, fbo_size) :
        self.prg.shading(
            queue=self.queue, 
            global_size=fbo_size, 
            local_size=None, 
            arg0=frame_buffer
        )
