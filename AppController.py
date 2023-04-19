import functools as ft
import glfw
#import glm
import Matrix as mat
import logging
import numpy as np
from OpenGL.GL import *
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties

import Raycaster
import Renderer
import VolumeData

class AppController :
    def __init__(self, setting) :
        self.setting=setting
        self.__init_window__()
        self.__init_cl__()

        # make rederer
        self.renderer = Renderer.Renderer(setting["RAY_DOMAIN"])


        self.gl_progs = []
        for shader_name, shader_file in setting["SHADER"].items() : 
            prg = Renderer.GLProgram(shader_file[0], shader_file[1])
            self.gl_progs.append((shader_name, prg))
        self.current_prog = 0
        self.gl_prog = self.gl_progs[self.current_prog][1]
        self.gl_prog.use()
            
        gl_texture = self.renderer.gen_deffered_textures(self.setting["RAY_DOMAIN"])
        self.renderer.gen_colormap()

        # make frame buffers
        self.deffered_buffer = [cl.GLTexture(self.ctx, cl.mem_flags.READ_WRITE, GL_TEXTURE_2D, 0, tex_id, 2) 
                                    for tex_id in gl_texture]

        # make raycaster with various kernels
        self.raycasters = []
        with open('./kernel/raycaster.cl', 'r') as fp : base_src = fp.read()
        for kernel_name, (kernel_file, qi_coeff) in setting["SPLINE_KERNEL"].items() : 
            with open(kernel_file, 'r') as fp : 
                raycaster_src = base_src+fp.read()
                raycaster = Raycaster.Raycaster(self.ctx, 
                                                self.devices, 
                                                self.queue, 
                                                raycaster_src, 
                                                self.setting["RAY_DOMAIN"])

                self.raycasters.append(((kernel_name, qi_coeff), raycaster))
        self.current_kernel = 0
        self.raycaster = self.raycasters[self.current_kernel][1]
        
        # load volume data
        self.volume_datas = []
        for data_name, (FILE_PATH, dim, scl, typ, ort, iso) in self.setting["VOLUME_DATA"].items() :
            volume_data = VolumeData.VolumeData(self.ctx, self.devices, self.queue)
            volume_data.uploadVolumeData(FILE_PATH, dim, typ, ort, scl)
            volume_data.applyQuasiInterpolator(self.raycasters[self.current_kernel][0][1])
            self.volume_datas.append(((data_name, dim, iso), volume_data))

        self.current_data = 0
        self.volume_data = self.volume_datas[self.current_data][1]
        self.isovalue = self.volume_datas[self.current_data][0][2]
        self.withQI = False
        # window setting
        w,h = glfw.get_framebuffer_size(self.wnd)
        self.callback_resize(self.wnd, w, h)

        self.fov = self.setting["FOV"]
        
        d2r = lambda th : th/180*np.pi

        # set default Model matrix
        Model = np.eye(4, dtype=np.float32)
        
        # set View matrix
        View = mat.lookAt((0,0,1+1/np.tan(d2r(self.fov/2))), (0,0,0), (0,1,0))

        # set Projection matrix
        Projection = mat.perspective(d2r(self.fov), 1, 1/np.tan(d2r(self.fov/2)), 2+1/np.tan(d2r(self.fov/2)))
        
        self.__update_MVP(Model, View, Projection) 


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
        vol_data = self.volume_data.getVolumeData(self.withQI)
        
        cl.enqueue_acquire_gl_objects(self.queue, self.deffered_buffer)
        evt1 = self.raycaster.genRay(self.invMVP, self.volume_data.data_ratio)
        evt2 = self.raycaster.raycast(self.isovalue, 
                                      self.deffered_buffer[0], 
                                      vol_data,
                                      self.volume_data.data_ratio)
        evt3 = self.raycaster.evalGradient(self.deffered_buffer[1], 
                                           self.deffered_buffer[0], 
                                           vol_data,
                                           self.volume_data.data_ratio)
        evt4 = self.raycaster.evalHessian(self.deffered_buffer[2], 
                                          self.deffered_buffer[3], 
                                          self.deffered_buffer[0], 
                                          vol_data,
                                          self.volume_data.data_ratio)
        cl.enqueue_release_gl_objects(self.queue, self.deffered_buffer)
        self.queue.finish()
        return (evt1, evt2, evt3, evt4)


    def rendering(self) :
        self.renderer.rendering()

    def mainloop(self) :
        frameCount = 0
        lastTime = 0
        #evt = []
        while not glfw.window_should_close(self.wnd):
            #evt.append(self.update())
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
                #msec = (lambda evt:(evt.profile.end-evt.profile.start)*1E-6)

                #evt = np.array([(msec(e1),msec(e2),msec(e3),msec(e4)) for e1, e2, e3, e4 in evt])
                #noe = len(evt)
                #evt = np.sum(evt, axis=0)
                #evt = evt/noe

                #Log.debug("Ray Generator : {0:.4f} msec".format(evt[0]))
                #Log.debug("Raycasting    : {0:.4f} msec".format(evt[1]))
                #Log.debug("Gradient      : {0:.4f} msec".format(evt[2]))
                #Log.debug("Hessian       : {0:.4f} msec".format(evt[3]))

                glfw.set_window_title(self.wnd, "Renderer({0:.2f} fps)".format(fps))
                frameCount = 0
                lastTime   = currentTime
                #evt = []

        glfw.terminate()

    def callback_resize(self, window, w, h ) :
        self.current_fbo_size = glfw.get_framebuffer_size(window)
        glViewport(0, 0, self.current_fbo_size[0], self.current_fbo_size[1])

    def __update_MVP(self, M, V, P) :
        self.Model = M
        self.View = V
        self.Projection = P
        self.MV = np.dot(V,M)
        self.MVP = np.dot(P,self.MV)
        self.invMVP = np.linalg.inv(self.MVP)
        self.gl_prog.update_uniform({"MV":self.MV})

    def callback_keyboard(self, window, key, scancode, action, mods) :
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS :
            glfw.set_window_should_close(window, GL_TRUE);

        if action == glfw.PRESS :
            match key :
                # Change isovalue
                case glfw.KEY_EQUAL :
                    self.isovalue += 0.01;
                    Log.info("Current Isovalue : {0}".format(self.isovalue))

                case glfw.KEY_MINUS :
                    self.isovalue -= 0.01;
                    Log.info("Current Isovalue : {0}".format(self.isovalue))

                # Select kernel
                case glfw.KEY_K :
                    self.current_kernel += 1
                    if self.current_kernel >= len(self.raycasters) :
                        self.current_kernel = 0
                    self.volume_data.applyQuasiInterpolator(self.raycasters[self.current_kernel][0][1])
                    self.raycaster = self.raycasters[self.current_kernel][1]
                    Log.info("Current Kernel : {0}".format(self.raycasters[self.current_kernel][0][0]))

                # Use quasi interpolator
                case glfw.KEY_Q :
                    self.withQI = not self.withQI

                # Select shader
                case glfw.KEY_S :
                    self.current_prog += 1
                    if self.current_prog >= len(self.gl_progs) :
                        self.current_prog = 0
                    self.gl_prog = self.gl_progs[self.current_prog][1]
                    self.gl_prog.use()
                    self.__update_MVP(self.Model, self.View, self.Projection)
                    self.gl_prog.update_uniform({"orientation":np.int32(self.volume_data.orientation)})
                    Log.info("Current Shader : {0}".format(self.gl_progs[self.current_prog][0]))

                # Select volume data
                case glfw.KEY_V :
                    self.current_data += 1
                    if self.current_data >= len(self.volume_datas) :
                        self.current_data = 0
                    self.volume_data = self.volume_datas[self.current_data][1]
                    self.volume_data.applyQuasiInterpolator(self.raycasters[self.current_kernel][0][1])

                    self.isovalue = self.volume_datas[self.current_data][0][2]
                    self.gl_prog.update_uniform({"orientation":np.int32(self.volume_data.orientation)})
                    Log.info("Current Volume Data : {0}/{1}".format(self.volume_datas[self.current_data][0][0],self.volume_data.orientation))

                    
    def __to_arcball_coordinate(self, pos) :
        pos = ((pos[0]-256)/256, (256-pos[1])/256)
        return (pos[0], pos[1], (2**2-(pos[0]**2 + pos[1]**2))**0.5);
    
    def __rotate_arcball(self, last_pos, cur_pos) :
        axis = np.cross(last_pos, cur_pos)
        axis = axis / np.linalg.norm(axis)
        th = np.pi/2-np.arccos( np.linalg.norm((np.array(cur_pos)-np.array(last_pos)))/2)
        
        if np.any(np.isnan(axis)) or np.isnan(th) :
            M = np.eye(4)
        else : 
            M = mat.rotate(tuple(axis), th)
        
        return M
        
    def callback_mouse(self, window, btn, act, mods) :
        if btn == glfw.MOUSE_BUTTON_LEFT :
            if(act == glfw.PRESS) :
                self.__last_cursor_pos = glfw.get_cursor_pos(window)
                self.__last_arcball_pos = self.__to_arcball_coordinate(self.__last_cursor_pos)
                self.__last_Model = self.Model

    def callback_cursor_position(self, window, xpos, ypos) :
        check_key = lambda key :any([glfw.get_key(window, k)==glfw.PRESS for k in key])
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS :
            current_pos = glfw.get_cursor_pos(window)
            
            if check_key([glfw.KEY_RIGHT_SHIFT, glfw.KEY_LEFT_SHIFT]) : # Translate
                t = [current_pos[0]-self.__last_cursor_pos[0],
                    -current_pos[1]+self.__last_cursor_pos[1]]
                M = mat.translate([t[0]/100,t[1]/100,0])

            elif check_key([glfw.KEY_RIGHT_CONTROL, glfw.KEY_LEFT_CONTROL]) : # Scale
                s = self.__last_cursor_pos[1]/current_pos[1]
                M = mat.scale([s,s,s])

            else : # Rotation
                current_arcball_pos = self.__to_arcball_coordinate(current_pos)
                M = self.__rotate_arcball(self.__last_arcball_pos, current_arcball_pos)
       
            self.__update_MVP(np.dot(M,self.__last_Model), self.View, self.Projection)

    def callback_scroll(self, window, xoffset, yoffset) :
        old_fov = self.fov
        self.fov += yoffset
        if self.fov<0 : self.fov = 0
        if self.fov>90 : self.fov = 90

        if not old_fov == self.fov :
            Log.info("Current FOV : {0:.2f}".format(self.fov) if self.fov>0 else "Orthogonal Porjection")
        else :
            return

        d2r = lambda th : th/180*np.pi
        # set View matrix
        if self.fov == 0 :
            View = mat.lookAt((0,0,2), (0,0,0), (0,1,0))
            Projection = mat.ortho(-1, 1, -1, 1, -1, 1)
        else :
            View = mat.lookAt((0,0,1+1/np.tan(d2r(self.fov/2))), (0,0,0), (0,1,0))
            Projection = mat.perspective(d2r(self.fov), 1, 1/np.tan(d2r(self.fov/2)), 2+1/np.tan(d2r(self.fov/2)))

        self.__update_MVP(self.Model, View, Projection)


if __name__ == "__main__":
    Log = logging.getLogger("Raycaster")
    Log.setLevel(logging.DEBUG)
    hFileLog = logging.FileHandler("./output/raycaster.log")
    hStreamLog = logging.StreamHandler()
    formatter = logging.Formatter(fmt='[%(levelname)s][%(asctime)s.%(msecs)03d][%(funcName)s():%(lineno)d] %(message)s', 
                              datefmt='%H:%M:%S')
    hFileLog.setFormatter(formatter)
    hStreamLog.setFormatter(formatter)
    Log.addHandler(hFileLog)
    Log.addHandler(hStreamLog)
    
    setting = {
        "WIN_WIDTH" : 512,
        "WIN_HEIGHT" : 512,
        "RAY_DOMAIN" : (512, 512),
        "VOLUME_DATA" : {"ML_40": ["./data/ML_40_O.raw", (40, 40, 40, 1), (1,1,1,1), np.float32, 1, 0.5],
                         "ML_80": ["./data/ML_80_O.raw", (80, 80, 80, 1), (1,1,1,1), np.float32, 1, 0.5],
                         "Dragon": ["./data/Dragon_256_O.raw", (256,256,128, 1), (1,1,1,1), np.float32, -1, 0],
                         'Carp'  : ["./data/Carp.raw", (256,256,512), (0.78125,0.390625,1,1), '>i2', 1, 1200],
                         },
        "SHADER" : {"Blinn-Phong" : ["./shader/default.vsh", "./shader/Blinn_Phong.fsh"],
                    "Min/Max-Curvature" : ["./shader/default.vsh", "./shader/MinMaxCurvature.fsh"],
                    #"Dxx" : ["./shader/default.vsh", "./shader/Dxx.fsh"],
                    #"Dyy" : ["./shader/default.vsh", "./shader/Dyy.fsh"],
                    #"Dzz" : ["./shader/default.vsh", "./shader/Dzz.fsh"],
                    #"Dyz" : ["./shader/default.vsh", "./shader/Dyz.fsh"],
                    #"Dzx" : ["./shader/default.vsh", "./shader/Dzx.fsh"],
                    #"Dxy" : ["./shader/default.vsh", "./shader/Dxy.fsh"]
                    },
        "SPLINE_KERNEL" : {"Six Direction Box-Spline on CC" : ["./kernel/cc6.cl",    (2.0, -1/6, 0.0, 0.0)], 
                           "Second Order FCC Voronoi-Spline": ["./kernel/fcc_v2.cl", (1.0, 0.0, 0.0, 0.0)], 
                           "Third Order FCC Voronoi-Spline" : ["./kernel/fcc_v3.cl", (3/2, 0.0, -1/24, 0.0)]},
        "FOV" : 45,
    }

    ctrl = AppController(setting)
    ctrl.mainloop()

    exit()
