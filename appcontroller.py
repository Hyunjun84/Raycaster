"""
raycaster.py

# Copyright (c) 2023, Hyunjun Kim
# All rights reserved.

"""
import configparser
import functools as ft
import glfw
import itertools as it
import logging
import numpy as np
from OpenGL.GL import *
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
from PIL import Image
import helper_matrix as mat
import raycaster
import renderer
import volumedata
from lattice import Lattice

class AppController :
    def __init__(self, config_file="config.ini") :
        config = configparser.ConfigParser()
        #config.optionxform = str
        config.read(config_file)
        config = config["DEFAULT"]

        self.__initWindow__(eval(config["window size"]))
        self.__initCL__()

        # make rederer
        self.renderer = renderer.Renderer(eval(config["ray domain"]))

        # load shader
        shader_config = configparser.ConfigParser()
        shader_config.read(config["shader"])
        self.pool_shader = []
        for shader_name in shader_config.sections() :
            prg = renderer.GLProgram(shader_name,
                                     shader_config[shader_name]["vertex shader"],
                                     shader_config[shader_name]["fragment shader"])
            self.pool_shader.append(prg)
        self.pool_shader = it.cycle(self.pool_shader)
        self.shader = next(self.pool_shader)
        self.shader.use()

        # make deffered buffer
        gl_texture = self.renderer.genDefferedTextures(eval(config["ray domain"]))
        self.renderer.genColormap()

        # get deffered buffer handle
        self.deffered_buffer = [cl.GLTexture(self.ctx, cl.mem_flags.READ_WRITE, GL_TEXTURE_2D, 0, tex_id, 2)
                                    for tex_id in gl_texture]

        # load raycaster with various kernels
        kernel_config = configparser.ConfigParser()
        kernel_config.read(config["kernel"])
        self.pool_raycaster = []
        with open('./kernel/raycaster.cl', 'r') as fp :
            base_src = fp.read()
            for kernel_name in kernel_config.sections() :
                with open(kernel_config[kernel_name]["path"], "r") as fp :
                    raycaster_src = base_src+fp.read()
                    r = raycaster.Raycaster(self.ctx,
                                            self.devices,
                                            self.queue,
                                            kernel_name,
                                            raycaster_src,
                                            eval(config["ray domain"]),
                                            eval(kernel_config[kernel_name]['sampling lattice']),
                                            eval(kernel_config[kernel_name]['quasi coefficients']))
                    self.pool_raycaster.append(r)
        self.pool_raycaster = it.cycle(self.pool_raycaster)
        self.raycaster = next(self.pool_raycaster)

        # load volume data
        dataset_config = configparser.ConfigParser()
        dataset_config.read(config["dataset"])
        self.pool_dataset = []
        for s in dataset_config.sections() :
            volume_data = volumedata.VolumeData(self.ctx, self.devices, self.queue)
            volume_data.uploadVolumeDataWithConfig(dataset_config[s])
        

            self.pool_dataset.append(volume_data)

        self.pool_dataset = it.cycle(self.pool_dataset)
        self.volume_data = next(self.pool_dataset)
        self.volume_data.applyQuasiInterpolator(self.raycaster.qi_coeff)
        self.isovalue = self.volume_data.isovalue

        self.withQI = False

        # window setting
        w,h = glfw.get_framebuffer_size(self.wnd)
        self.__callbackResize(self.wnd, w, h)

        self.fov = np.clip(float(config["FOV"]), 0, 90)

        # set default Model matrix
        m = self.volume_data.model

        # set View and Projection matrix
        if self.fov == 0 :
            v = mat.lookAt((0,0,2), (0,0,0), (0,1,0))
            p = mat.ortho(-1, 1, -1, 1, -1, 1)
        else :
            d2r = lambda th : th/180*np.pi
            cotan_th = 1/np.tan(d2r(self.fov/2))
            v = mat.lookAt((0,0,cotan_th), (0,0,0), (0,1,0))
            p = mat.perspective(d2r(self.fov), 1, cotan_th, 2+cotan_th)

        self.use_global_transform = eval(config["use global transform"])
        self.updateMVP(m, v, p)
        self.shader.setUniform({"orientation":np.int32(self.volume_data.orientation)})


    def __initCL__(self) :
        self.platforms = cl.get_platforms()
        self.devices = self.platforms[0].get_devices(device_type=cl.device_type.GPU)
        ctx_properties = [(cl.context_properties.PLATFORM, self.platforms[0])]
        ctx_properties = ctx_properties + get_gl_sharing_context_properties()
        self.ctx = cl.Context(dev_type=cl.device_type.GPU,
                              properties=ctx_properties)
        self.queue = cl.CommandQueue(context=self.ctx, device=self.devices[0], 
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    def __initWindow__(self, window_size) :
        if not glfw.init(): return False

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        self.wnd = glfw.create_window(
                        window_size[0],
                        window_size[1],
                        "Renderer",
                        None, None)

        if not self.wnd :
            glfw.terminate()
            return False

        glfw.make_context_current(self.wnd)
        glfw.swap_interval(0)

        glfw.set_framebuffer_size_callback(self.wnd, self.__callbackResize)
        glfw.set_key_callback(self.wnd, self.__callbackKeyboard)
        glfw.set_mouse_button_callback(self.wnd, self.__callbackMouse)
        glfw.set_scroll_callback(self.wnd, self.__callbackScroll)
        glfw.set_cursor_pos_callback(self.wnd, self.__callbackCursorPosition)

        return True

    def update(self) :
        vol_data = self.volume_data.getVolumeData(self.withQI)

        cl.enqueue_acquire_gl_objects(self.queue, self.deffered_buffer)
        evt1 = self.raycaster.genRay(self.inv_mvp, self.volume_data.data_ratio)
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
        frame_count = 0
        last_time = 0
        #evt = []
        while not glfw.window_should_close(self.wnd):
            #evt.append(self.update())
            self.update()
            self.rendering()

            glfw.swap_buffers(self.wnd)
            glfw.poll_events()

            frame_count  = frame_count+1
            current_time = glfw.get_time()
            delta_time   = current_time - last_time

            if(delta_time >= 2.0) :
                fps = frame_count/delta_time
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
                frame_count = 0
                last_time   = current_time
                #evt = []

        glfw.terminate()

    def __callbackResize(self, window, w, h ) :
        self.current_fbo_size = glfw.get_framebuffer_size(window)
        glViewport(0, 0, self.current_fbo_size[0], self.current_fbo_size[1])

    def updateMVP(self, m, v, p) :
        self.model = m
        self.view = v
        self.projection = p
        self.mv = np.dot(v, m)
        self.mvp = np.dot(p, self.mv)
        self.inv_mvp = np.linalg.inv(self.mvp)
        self.shader.setUniform({"MV":self.mv})

    def __callbackKeyboard(self, window, key, scancode, action, mods) :
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
                    self.raycaster = next(self.pool_raycaster)
                    while self.volume_data.lattice != self.raycaster.lattice :
                        self.raycaster = next(self.pool_raycaster)
                    evt = self.volume_data.applyQuasiInterpolator(self.raycaster.qi_coeff)
                    #self.queue.finish()
                    #msec = (lambda evt:(evt.profile.end-evt.profile.start)*1E-6)
                    #Log.info("Apply Quasi prefilter : {0} msec.".format(msec(evt), "msec"))
                    Log.info("Current Kernel : {0}".format(self.raycaster.title))
                    Log.info("Current Volume Data : {0}/{1}".format(self.volume_data.title, self.volume_data.orientation))

                # Use quasi interpolator
                case glfw.KEY_Q :
                    self.withQI = not self.withQI

                # Save screenshot
                case glfw.KEY_X :
                    buf = self.renderer.getFrameBufferData()
                    img = Image.frombytes('RGBA', (512, 512), buf).transpose(Image.FLIP_TOP_BOTTOM)
                    img.save("screenshot/{0}_{1}.png".format(self.volume_data.title, self.raycaster.title))

                # Select shader
                case glfw.KEY_S :
                    self.shader = next(self.pool_shader)
                    self.shader.use()
                    self.shader.setUniform({"orientation":np.int32(self.volume_data.orientation)})
                    self.updateMVP(self.model, self.view, self.projection)
                    Log.info("Current Shader : {0}".format(self.shader.title))
                    
                # Select volume data
                case glfw.KEY_V :
                    self.volume_data.model = self.model
                    self.volume_data = next(self.pool_dataset)
                    self.isovalue = self.volume_data.isovalue
                    
                    current_raycaster = self.raycaster.title
                    while self.raycaster.lattice != self.volume_data.lattice :
                        self.raycaster = next(self.pool_raycaster)
                        if self.raycaster.title == current_raycaster.title : 
                            exit(1)

                    self.volume_data.applyQuasiInterpolator(self.raycaster.qi_coeff)

                    self.shader.use()
                    self.shader.setUniform({"orientation":np.int32(self.volume_data.orientation)})

                    m = self.model if self.use_global_transform else self.volume_data.model
                    
                    self.updateMVP(m, self.view, self.projection)
                    Log.info("Current Volume Data : {0}/{1}".format(self.volume_data.title, self.volume_data.orientation))
                    Log.info("Current Kernel : {0}".format(self.raycaster.title))

                # Print current status
                case glfw.KEY_I :
                    Log.info("====================================================")
                    Log.info("Status")
                    Log.info("\tKernel : {0}{1}".format(self.raycaster.title, " with Quasi Interpolation" if self.withQI==1 else ""))
                    Log.info("\tDataset : {0}".format(self.volume_data.title))
                    Log.info("\tShader : {0}".format(self.shader.title))
                    Log.info("\tisovalue : {0}".format(self.isovalue))
                    Log.info("\tModel matrix : {0}".format(self.model.tolist()))
                    Log.info("\tView matrix : {0}".format(self.view.tolist()))
                    Log.info("\tProjection matrix : {0}".format(self.projection.tolist()))

    def __toArcballCoordinates__(self, pos) :
        pos = ((pos[0]-256)/256, (256-pos[1])/256)
        return (pos[0], pos[1], (4-(pos[0]**2 + pos[1]**2))**0.5);

    def __rotateArcball__(self, last_pos, cur_pos, is_click=False) :
        axis_click = 0.2
        angle_click = 15 # degree
        delta = np.array(last_pos) - np.array(cur_pos)
        
        th = np.pi/2-np.arccos( np.linalg.norm(delta)/2)
        if is_click :
            mag = np.abs(delta)
            if mag[0]>mag[1] :
                axis = np.array([0, -np.sign(delta[0]), 0])
            else :
                axis = np.array([np.sign(delta[1]), 0, 0])
            th = ((th/np.pi*180)//angle_click*angle_click)/180*np.pi
        else : 
            axis = np.cross(last_pos, cur_pos)
        
        if np.linalg.norm(axis) == 0 : return np.eye(4, dtype=np.float32)

        axis = axis / np.linalg.norm(axis)
    
        if np.any(np.isnan(axis)) or np.isnan(th) :
            M = np.eye(4, dtype=np.float32)
        else :
            M = mat.rotate(tuple(axis), th)

        return M

    def __callbackMouse(self, window, btn, act, mods) :
        if btn == glfw.MOUSE_BUTTON_LEFT :
            if(act == glfw.PRESS) :
                self.__last_cursor_pos = glfw.get_cursor_pos(window)
                self.__last_arcball_pos = self.__toArcballCoordinates__(self.__last_cursor_pos)
                self.__last_model = self.model

    def __callbackCursorPosition(self, window, xpos, ypos) :
        check_key = lambda key :any([glfw.get_key(window, k)==glfw.PRESS for k in key])
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS :
            current_pos = glfw.get_cursor_pos(window)

            if check_key([glfw.KEY_RIGHT_SHIFT, glfw.KEY_LEFT_SHIFT]) : # Translate
                t = [current_pos[0]-self.__last_cursor_pos[0],
                    -current_pos[1]+self.__last_cursor_pos[1]]
                m = mat.translate([t[0]/100,t[1]/100,0])

            elif check_key([glfw.KEY_RIGHT_CONTROL, glfw.KEY_LEFT_CONTROL]) : # Scale
                s = self.__last_cursor_pos[1]/current_pos[1]
                m = mat.scale([s,s,s])

            else : # Rotation
                is_click = check_key([glfw.KEY_RIGHT_ALT, glfw.KEY_LEFT_ALT])
                current_arcball_pos = self.__toArcballCoordinates__(current_pos)
                m = self.__rotateArcball__(self.__last_arcball_pos, current_arcball_pos, is_click)

            self.updateMVP(np.dot(m, self.__last_model), self.view, self.projection)

    def __callbackScroll(self, window, xoffset, yoffset) :
        old_fov = self.fov
        self.fov = np.clip(self.fov+yoffset, 0, 90)

        if not old_fov == self.fov :
            Log.info("Current FOV : {0:.2f}".format(self.fov) if self.fov>0 else "Orthogonal Porjection")
        else :
            return

        if self.fov == 0 :
            v = mat.lookAt((0,0,10), (0,0,0), (0,1,0))
            p = mat.ortho(-1, 1, -1, 1, -1, 1)
        else :
            d2r = lambda th : th/180*np.pi
            cotan_th = 1/np.tan(d2r(self.fov/2))
            v = mat.lookAt((0,0,cotan_th), (0,0,0), (0,1,0))
            p = mat.perspective(d2r(self.fov), 1, cotan_th, 2+cotan_th)
     
        self.updateMVP(self.model, v, p)


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

    ctrl = AppController("config.ini")
    ctrl.mainloop()

    exit()
