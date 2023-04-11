import numpy as np
from OpenGL.GL  import *

class Renderer :
    def __init__(self, fbo_size) :
        self.__init_quad__()
        self.fbo_size = [0,0]
        self.gen_deffered_textures(fbo_size)
        self.gen_colormap()

    def gen_deffered_textures(self, fbo_size) :
        #if self.fbo_size == fbo_size : return self.texid
        internal_format = GL_RGBA32F
        format = GL_RGBA
        self.texid = glGenTextures(4)
        for i in range(4) :
            glActiveTexture(GL_TEXTURE0+i)
            glBindTexture(GL_TEXTURE_2D, self.texid[i])
            glPixelStorei(GL_UNPACK_ALIGNMENT,1)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, 
                fbo_size[0], fbo_size[1], 0, 
                format, GL_FLOAT, np.zeros([fbo_size[0], fbo_size[1], 4]).astype(np.float32))

        
        return self.texid

    def gen_colormap(self) :
         # colormap for min-max curvature
        colormap = np.array([[ 1, 0, 0], [ 1, 1, 0], [0,1,0],
                             [.5,.5,.5], [.5,.5,.5], [0,1,1],
                             [.5,.5,.5], [.5,.5,.5], [0,0,1]], dtype=np.float32)

        self.tex_colormap = glGenTextures(1)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, self.tex_colormap)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3, 3, 0, GL_RGB, GL_FLOAT, colormap)
        #glBindTexture(GL_TEXTURE_2D, self.tex_colormap)
        return self.tex_colormap

    def __init_quad__(self) :
        quad = np.array([
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 0.0, 1.0],
            [ 1.0,  1.0, 0.0, 1.0],
            [-1.0,  1.0, 0.0, 1.0],
            ]).astype(np.float32)

        idx = np.array([
            [0, 1, 3],
            [1, 2, 3],
            ]).astype(np.uint32)

        self.__VAO = glGenVertexArrays(1)
        self.__VBO = glGenBuffers(2)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.__VBO[0])
        glBufferData(GL_ARRAY_BUFFER, quad, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__VBO[1])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glBindVertexArray(self.__VAO)

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.__VBO[0])
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__VBO[1])
        
        glBindVertexArray(0)

    def rendering(self) :
        glClear(GL_COLOR_BUFFER_BIT)
        glBindVertexArray(self.__VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glFinish()

class GLProgram :
    def __init__(self, vtx_shader, frag_shader) :
        self.__create_program__(vtx_shader, frag_shader)
        self.use()
        glUniform1i(self.uniforms['tex_position'], 0)
        glUniform1i(self.uniforms['tex_gradient'], 1)
        glUniform1i(self.uniforms['tex_HessianII'], 2)
        glUniform1i(self.uniforms['tex_HessianIJ'], 3)
        glUniform1i(self.uniforms['tex_colormap'], 4)
        

        self.update_uniform(np.eye(4).astype(np.float32))


    def __create_program__(self, vtx_shader, frag_shader) :
        self.prg = glCreateProgram()
        vsh = self.loadShader(vtx_shader, GL_VERTEX_SHADER)
        fsh = self.loadShader(frag_shader, GL_FRAGMENT_SHADER)
        glAttachShader(self.prg, vsh)
        glAttachShader(self.prg, fsh)
        glLinkProgram(self.prg)

        success = glGetProgramiv(self.prg, GL_LINK_STATUS)
        if not success :  
            print(glGetProgramInfoLog(self.prg,))
            exit(0)

        glDeleteShader(vsh)
        glDeleteShader(fsh)

        self.uniforms = {"MV" : glGetUniformLocation(self.prg, 'MV'),
                         "tex_position" : glGetUniformLocation(self.prg, 'tex_position'),
                         "tex_gradient" : glGetUniformLocation(self.prg, 'tex_gradient'),
                         "tex_HessianII" : glGetUniformLocation(self.prg, 'tex_HessianII'),
                         "tex_HessianIJ" : glGetUniformLocation(self.prg, 'tex_HessianIJ'),
                         "tex_colormap" : glGetUniformLocation(self.prg, 'tex_colormap'),
                         "orientation" : glGetUniformLocation(self.prg, 'orientation'),
                         }
        
    def loadShader(self, filename, shd_type) :
        with open(filename, 'r') as fp : src=fp.read()
        shd = glCreateShader(shd_type)
    
        glShaderSource(shd, (src, ), None)
        glCompileShader(shd)
        success = glGetShaderiv(shd, GL_COMPILE_STATUS)
        if not success :
            print(glGetShaderInfoLog(shd))
            return -1;
        return shd

    def use(self) :
        glUseProgram(self.prg)

    def setDataOrientation(self, orientation) :
        glUniform1i(self.uniforms["orientation"], np.int32(orientation))
    
    def update_uniform(self, MV) :
        glUniformMatrix4fv(self.uniforms["MV"], 1, GL_FALSE, np.float32(MV))

