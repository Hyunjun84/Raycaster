import pyopencl as cl

class ColorMapper :
    def __init__(self, ctx, devices, queue) :
    self.ctx = ctx
    self.queue = queue
    with open('./kernel/mandelbrot.cl', 'r') as fp : src = fp.read()
    self.prg = cl.Program(self.ctx, src)
    self.prg.build(options=["-g",],devices=[devices[0],], cache_dir=None)
    self.bound = [[-2,1],[-1.5,1.5]]
    self.setFBOSize([1024,1024])
