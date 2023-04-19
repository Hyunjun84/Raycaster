#version 330 core

in vec2 vTexCoord;

out vec4 fColor;

uniform sampler2D tex_position;
uniform sampler2D tex_gradient;
//uniform sampler2D tex_colormap;
uniform mat4        MV;
uniform int orientation;

struct TMaterial
{
	vec3	ambient;
	vec3	diffuse;
	vec3	specular;
	vec3	emission;
	float	shininess;
};
struct TLight
{
	vec4	position;
	vec3	ambient;
	vec3	diffuse;
	vec3	specular;
};

/*
http://devernay.free.fr/cours/opengl/materials.html
emerald	0.0215	0.1745	0.0215	0.07568	0.61424	0.07568	0.633	0.727811	0.633	0.6
jade	0.135	0.2225	0.1575	0.54	0.89	0.63	0.316228	0.316228	0.316228	0.1
obsidian	0.05375	0.05	0.06625	0.18275	0.17	0.22525	0.332741	0.328634	0.346435	0.3
pearl	0.25	0.20725	0.20725	1	0.829	0.829	0.296648	0.296648	0.296648	0.088
ruby	0.1745	0.01175	0.01175	0.61424	0.04136	0.04136	0.727811	0.626959	0.626959	0.6
turquoise	0.1	0.18725	0.1745	0.396	0.74151	0.69102	0.297254	0.30829	0.306678	0.1
brass	0.329412	0.223529	0.027451	0.780392	0.568627	0.113725	0.992157	0.941176	0.807843	0.21794872
bronze	0.2125	0.1275	0.054	0.714	0.4284	0.18144	0.393548	0.271906	0.166721	0.2
chrome	0.25	0.25	0.25	0.4	0.4	0.4	0.774597	0.774597	0.774597	0.6
copper	0.19125	0.0735	0.0225	0.7038	0.27048	0.0828	0.256777	0.137622	0.086014	0.1
gold	0.24725	0.1995	0.0745	0.75164	0.60648	0.22648	0.628281	0.555802	0.366065	0.4
silver	0.19225	0.19225	0.19225	0.50754	0.50754	0.50754	0.508273	0.508273	0.508273	0.4
black plastic	0.0	0.0	0.0	0.01	0.01	0.01	0.50	0.50	0.50	.25
cyan plastic	0.0	0.1	0.06	0.0	0.50980392	0.50980392	0.50196078	0.50196078	0.50196078	.25
green plastic	0.0	0.0	0.0	0.1	0.35	0.1	0.45	0.55	0.45	.25
red plastic	0.0	0.0	0.0	0.5	0.0	0.0	0.7	0.6	0.6	.25
white plastic	0.0	0.0	0.0	0.55	0.55	0.55	0.70	0.70	0.70	.25
yellow plastic	0.0	0.0	0.0	0.5	0.5	0.0	0.60	0.60	0.50	.25
black rubber	0.02	0.02	0.02	0.01	0.01	0.01	0.4	0.4	0.4	.078125
cyan rubber	0.0	0.05	0.05	0.4	0.5	0.5	0.04	0.7	0.7	.078125
green rubber	0.0	0.05	0.0	0.4	0.5	0.4	0.04	0.7	0.04	.078125
red rubber	0.05	0.0	0.0	0.5	0.4	0.4	0.7	0.04	0.04	.078125
white rubber	0.05	0.05	0.05	0.5	0.5	0.5	0.7	0.7	0.7	.078125
yellow rubber	0.05	0.05	0.0	0.5	0.5	0.4	0.7	0.7	0.04	.078125
*/

TMaterial uMaterial[2] =
	TMaterial[2]
	(
	    TMaterial(
	    	vec3(0.135, 0.2225, 0.1575), 
	    	vec3(0.54,0.89,0.63), 
	    	vec3(0.316228,0.316228,0.316228), 
	    	vec3(0,0,0), 
	    	0.1*128),
	    TMaterial(
			vec3(0, 0, 0),
	        vec3(0.5, 0.0, 0.0),
	        vec3(0.7, 0.6, 0.6),
			vec3(0,0,0),
	        .25*128.0)
	);

TLight uLight = TLight(
        vec4(-10,10,10,0),
        vec3(.8,.8,.8),
        vec3(1,1,1),
        vec3(1,1,1)
        );

vec4 shade_Blinn_Phong(vec3 n, vec4 pos_eye, TMaterial material, TLight light)
{
	vec3	l;
	if(light.position.w == 1.0)
		l = normalize((light.position - pos_eye).xyz);		// positional light
	else
		l = normalize((light.position).xyz);	// directional light

	vec3	v = -normalize(pos_eye.xyz);
	vec3	h = normalize(l + v);
	float	l_dot_n = max(dot(l, n), 0.0);
	vec3	ambient = light.ambient * material.ambient;
	vec3	diffuse = light.diffuse * material.diffuse * l_dot_n;
	vec3	specular = vec3(0.0);

	if(l_dot_n >= 0.0) {
		specular = light.specular * material.specular * pow(max(dot(h, n), 0.0), material.shininess);
	}
	return vec4(ambient + diffuse + specular, 1);
}




void main() {
	vec4	p = texture(tex_position, vTexCoord);
	vec3	g = -normalize(texture(tex_gradient, vTexCoord).xyz);

	if(p.w != 0.0f) {
		fColor = shade_Blinn_Phong(normalize(mat3(MV)*(p.w*g)), MV*vec4(p.xyz,1), uMaterial[int(0.5*(1.0-p.w*orientation))], uLight);
	} else {
		fColor = vec4(0,0,0,1);
	}

}

