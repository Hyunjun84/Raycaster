import glfw
import logging
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import *
import numpy as np

import raycaster
import renderer

MAX_ITER = 2**30
class Controller :
    def __init__(self, setting) :
        self.setting=setting
        self.__init_window__()
        self.__init_cl__()
        self.renderer = renderer.Renderer()
        self.raycaster = raycaster.Raycaster(self.ctx, self.devices, self.queue)
        w,h = glfw.get_framebuffer_size(self.wnd)
        self.callback_resize(self.wnd, w, h)
        

    def __init_cl__(self) :        
        self.platforms = cl.get_platforms()
        self.devices = self.platforms[0].get_devices(device_type=cl.device_type.GPU)
        ctx_properties = [(cl.context_properties.PLATFORM, self.platforms[0])]
        ctx_properties = ctx_properties + get_gl_sharing_context_properties()
        self.ctx = cl.Context(dev_type=cl.device_type.GPU,
                              properties=ctx_properties)
        self.queue = cl.CommandQueue(context=self.ctx, 
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    def __init_window__(self) :
        if not glfw.init(): return False
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.wnd = glfw.create_window(
                        self.setting["WIN_WIDTH"], 
                        self.setting["WIN_HEIGHT"], 
                        "Renderer", 
                        None, None)

        if not self.wnd :
            glfw.terminate()
            return False

        glfw.make_context_current(self.wnd)
        glfw.swap_interval(0)
        glfw.set_framebuffer_size_callback(self.wnd, self.callback_resize)
        glfw.set_key_callback(self.wnd, self.callback_keyboard)
        glfw.set_mouse_button_callback(self.wnd, self.callback_mouse)
        glfw.set_scroll_callback(self.wnd, self.callback_scroll)

        return True

    def update(self) :

        cl.enqueue_acquire_gl_objects(self.queue, [self.frame_buffer,])
        self.raycaster.Shading(self.frame_buffer, [self.setting["FBO_WIDTH"], self.setting["FBO_HEIGHT"]])
        cl.enqueue_release_gl_objects(self.queue, [self.frame_buffer,])
        self.queue.finish()

    def rendering(self) :
        self.renderer.rendering()

    def mainloop(self) :
        frameCount = 0
        lastTime = 0
        while not glfw.window_should_close(self.wnd):
            self.update()
            self.rendering()

            glfw.swap_buffers(self.wnd)
            glfw.poll_events()

            frameCount  = frameCount+1
            currentTime = glfw.get_time()
            deltaTime   = currentTime - lastTime

            if(deltaTime >= 2.0) :
                Log.info("{0:.2f} FPS.".format(frameCount/deltaTime))
                frameCount = 0
                lastTime   = currentTime

        glfw.terminate()

    def callback_resize(self, window, w, h ) :
        print("current window size : ", w,h)
        fw,fh = glfw.get_framebuffer_size(window)
        fw = self.setting["FBO_WIDTH"]*w//self.setting["WIN_WIDTH"]
        fh = self.setting["FBO_HEIGHT"]*h//self.setting["WIN_HEIGHT"]

        print("target fbo size : ", fw,fh)

        
        gl_buffer = self.renderer.gen_frame_buffer_texture([fw,fh])
        self.frame_buffer = cl.GLTexture(self.ctx, cl.mem_flags.READ_WRITE, GL_TEXTURE_2D, 0, gl_buffer, 2)

        fw,fh = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fw, fh)



    
    def callback_keyboard(self, window, key, scancode, action, mods) :
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS :
            glfw.set_window_should_close(window, GL_TRUE);

    def callback_mouse(self, window, btn, act, mods) :
        Log.debug("btn/act/mod : {0}/{1}/{2}".format(btn,act,mods))
        

    def callback_scroll(self, window, xoffset, yoffset) :

        pass

if __name__ == "__main__":
    Log = logging.getLogger("Raycaster")
    Log.setLevel(logging.DEBUG)
    hFileLog = logging.FileHandler("./output/raycaster.log")
    hStreamLog = logging.StreamHandler()
    formatter = logging.Formatter(fmt='[%(levelname)s][%(asctime)s.%(msecs)03d][%(funcName)s():%(lineno)d] %(message)s', datefmt='%H:%M:%S')
    hFileLog.setFormatter(formatter)
    hStreamLog.setFormatter(formatter)
    Log.addHandler(hFileLog)
    Log.addHandler(hStreamLog)

    setting = {
        "WIN_WIDTH" : 1024,
        "WIN_HEIGHT" : 1024,
        "FBO_WIDTH" : 1024,
        "FBO_HEIGHT" : 1024,
    }

    ctrl = Controller(setting)
    ctrl.mainloop()

    exit()
