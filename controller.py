import glfw
import logging
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import *
import numpy as np

import raycaster
import renderer
import matrix

MAX_ITER = 2**30
class Controller :
    def __init__(self, setting) :
        self.setting=setting
        self.__init_window__()
        self.__init_cl__()

        self.isovalue = setting["ISOVALUE"]
        self.renderer = renderer.Renderer()
        gl_buffers = self.renderer.gen_deffered_textures(self.setting["RAY_DOMAIN"])

        flag = cl.mem_flags.READ_WRITE
        self.deffered_buffer = [cl.GLTexture(self.ctx, flag, GL_TEXTURE_2D, 0, buf, 2) for buf in gl_buffers]

        self.raycaster = raycaster.Raycaster(self.ctx, self.devices, self.queue, setting["SPLINE_KERNEL"])
        self.raycaster.setNumberofRays(self.setting["RAY_DOMAIN"])
        self.raycaster.uploadVolumeData(self.setting["VOLUME_DATA_PATH"], 
                                        self.setting["VOLUE_DATA_DIM"], 
                                        self.setting["VOLUE_DATA_TYPE"])

        w,h = glfw.get_framebuffer_size(self.wnd)
        self.callback_resize(self.wnd, w, h)
        View = np.eye(4).astype(np.float32)*0.75
        View[2,2] = -View[2,2]
        View[3][3] = 1

        self.__update_MVP(np.eye(4).astype(np.float32), View)
        self.renderer.update_uniform(self.MVP)
        self.fov = self.setting["FOV"]

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
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        
        self.wnd = glfw.create_window(
                        self.setting["WIN_WIDTH"], 
                        self.setting["WIN_HEIGHT"], 
                        "Renderer", 
                        None, None)

#        monitors = glfw.get_monitors()
#        workarea = glfw.get_monitor_workarea(monitors[1])
#        glfw.set_window_pos(self.wnd, workarea[0]+512, workarea[1]+256)

        if not self.wnd :
            glfw.terminate()
            return False

        glfw.make_context_current(self.wnd)
        glfw.swap_interval(0)

        glfw.set_framebuffer_size_callback(self.wnd, self.callback_resize)
        glfw.set_key_callback(self.wnd, self.callback_keyboard)
        glfw.set_mouse_button_callback(self.wnd, self.callback_mouse)
        glfw.set_scroll_callback(self.wnd, self.callback_scroll)
        glfw.set_cursor_pos_callback(self.wnd, self.callback_cursor_position)

        return True

    def update(self) :
        msec = (lambda evt:(evt.profile.end-evt.profile.start)*1E-6)
        cl.enqueue_acquire_gl_objects(self.queue, self.deffered_buffer)
        evt1 = self.raycaster.genRay(self.invMVP, np.float32(self.fov))
        #self.raycaster.ray_dump()
        #exit(0)
        evt2 = self.raycaster.raycast(self.isovalue, self.deffered_buffer[0])
        evt3 = self.raycaster.evalGradient(self.deffered_buffer[0], self.deffered_buffer[1])
        cl.enqueue_release_gl_objects(self.queue, self.deffered_buffer)
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
                fps = frameCount/deltaTime
                Log.info("{0:.2f} FPS.".format(fps))
                glfw.set_window_title(self.wnd, "Renderer({0:.2f} fps)".format(fps))
                frameCount = 0
                lastTime   = currentTime

        glfw.terminate()


    def callback_resize(self, window, w, h ) :
        self.current_fbo_size = glfw.get_framebuffer_size(window)
        glViewport(0, 0, self.current_fbo_size[0], self.current_fbo_size[1])

    def __update_MVP(self, M, V) :
        self.MVP = np.dot(M,V)
        self.Model = M
        self.View = V
        self.invMVP = np.linalg.inv(self.MVP)
        self.renderer.update_uniform(self.MVP)

    def callback_keyboard(self, window, key, scancode, action, mods) :
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS :
            glfw.set_window_should_close(window, GL_TRUE);

        if action == glfw.PRESS :
            th = np.pi/30

            match key :
                case glfw.KEY_EQUAL :
                    if mods == glfw.MOD_SHIFT :
                        self.isovalue += 0.01;
                        Log.info(self.isovalue)

                case glfw.KEY_K :
                    self.raycaster.nextKernel()

                case glfw.KEY_MINUS :
                    self.isovalue -= 0.01;
                    Log.info(self.isovalue)            

    def __to_arcball_coordinate(self, pos) :
        pos = ((pos[0]-256)/256, (256-pos[1])/256)
        return (pos[0], pos[1], (2**2-(pos[0]**2 + pos[1]**2))**0.5);
    
    def __rotate_arcball(self, last_pos, cur_pos) :
        axis = np.cross(last_pos, cur_pos)
        axis = axis / np.linalg.norm(axis)
        th = np.pi/2-np.arccos( np.linalg.norm((np.array(cur_pos)-np.array(last_pos)))/2)
        if np.any(np.isnan(axis)) or np.isnan(th) :
            return np.eye(4).astype(np.float32)
        return matrix.rotate(axis, th)

    def callback_mouse(self, window, btn, act, mods) :
        if btn == glfw.MOUSE_BUTTON_LEFT :
            if(act == glfw.PRESS) :
                self.__last_cursor_pos = self.__to_arcball_coordinate(glfw.get_cursor_pos(window))
                self.__last_Model = self.Model
                
            if(act == glfw.RELEASE) :
                try :
                    current_pos = self.__to_arcball_coordinate(glfw.get_cursor_pos(window))
                    M = self.__rotate_arcball(self.__last_cursor_pos, current_pos)
                    self.__update_MVP(np.dot(M, self.__last_Model), self.View )
                except :
                    Log.info("last cursor position is not defined.")


        elif btn == glfw.MOUSE_BUTTON_RIGHT :
            if(act == glfw.PRESS) :
                self.__last_cursor_pos = glfw.get_cursor_pos(window)
                self.__last_View = self.View
            
            if(act == glfw.RELEASE) :
                try :
                    current_pos = glfw.get_cursor_pos(window)
                    scale = self.__last_cursor_pos[1]/current_pos[1]
                    V = matrix.scale(scale, scale, scale)
                    self.__update_MVP(self.Model, np.dot(self.__last_View, V))
                except :
                    Log.info("last cursor position is not defined.")

#        Log.debug("btn/act/mod : {0}/{1}/{2}".format(btn,act,mods))

    def callback_cursor_position(self, window, xpos, ypos) :
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS :
            current_pos = self.__to_arcball_coordinate(glfw.get_cursor_pos(window))
            M = self.__rotate_arcball(self.__last_cursor_pos, current_pos)
            self.__update_MVP(np.dot(M, self.__last_Model), self.View )

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS :
            current_pos = glfw.get_cursor_pos(window)
            scale = self.__last_cursor_pos[1]/current_pos[1]
            V = matrix.scale(scale, scale, scale)
            self.__update_MVP(self.Model, np.dot(self.__last_View, V))

        
    def callback_scroll(self, window, xoffset, yoffset) :
        self.fov += yoffset
        if self.fov<0 : self.fov = 0
        if self.fov>90 : self.fov = 90

        Log.info("FOV : {0}".format(self.fov) if self.fov>0 else "Orthogonal Porjection")


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
        "WIN_WIDTH" : 512,
        "WIN_HEIGHT" : 512,
        "RAY_DOMAIN" : [512,512],
        "VOLUME_DATA_PATH" : "/Users/kamu/data/ML_20_O.raw",
        "VOLUE_DATA_DIM" : [20,20,20,1],
        "VOLUE_DATA_TYPE" : np.float32,
        "ISOVALUE" : 0.5,
        "SPLINE_KERNEL" : {"Six Direction Box-Spline on CC":"./kernel/cc6.cl", 
                           "Second Order FCC Voronoi-Spline":"./kernel/fcc_v2.cl", 
                           "Third Order FCC Voronoi-Spline":"./kernel/fcc_v3.cl"},
        "FOV" : 60,
    }

    ctrl = Controller(setting)
    ctrl.mainloop()

    exit()
