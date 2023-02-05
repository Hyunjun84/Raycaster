import numpy as np
from OpenGL.GL  import *

class Renderer :
    def __init__(self) :
        self.__create_program__()
        

        self.__init_quad__()
        self.gen_deffered_textures([512,512])

        
        glUseProgram(self.prg)
        glClearColor(0.0, 0.0, 0.0, 1.0)


        glUniform1i(self.uniforms['tex_position'], 0)
        glUniform1i(self.uniforms['tex_gradient'], 1)
        self.update_uniform(np.eye(4).astype(np.float32))

    def __del__(self) :
        pass
        #self.__release_quad()

    def gen_deffered_textures(self, fbo_size) :
        internal_format = GL_RGBA32F
        format = GL_RGBA

        self.texid = glGenTextures(2)

        for i in range(2) :
            glBindTexture(GL_TEXTURE_2D, self.texid[i])
            glPixelStorei(GL_UNPACK_ALIGNMENT,1)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, 
                fbo_size[0], fbo_size[1], 0, 
                format, GL_FLOAT, np.zeros([fbo_size[0], fbo_size[1], 4]).astype(np.float32))
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texid[0])
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.texid[1])

        return self.texid

    def update_uniform(self, MV) :
        glUniformMatrix4fv(self.uniforms["MV"], 1, GL_FALSE, np.float32(MV))        
        
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


    def __release_quad(self) :
        glDeleteVertexArrays(1, self.__VAO)
        glDeleteVertexBuffers(2, self.__VBO)        

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

    def __create_program__(self) :
        self.prg = glCreateProgram()
        vsh = self.loadShader("shader/default.vsh", GL_VERTEX_SHADER)
        fsh = self.loadShader("shader/Blinn_Phong.fsh", GL_FRAGMENT_SHADER)
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
                         "tex_gradient" : glGetUniformLocation(self.prg, 'tex_gradient'),}
        

    def rendering(self) :
        glClear(GL_COLOR_BUFFER_BIT)
        glBindVertexArray(self.__VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glFinish()
