#version 330 core

#define	SHADING_BLINN_PHONG	0

in vec2 vTexCoord;

out vec4 fColor;

uniform mat4        MV;

uniform sampler2D tex_position;
uniform sampler2D tex_gradient;
uniform sampler2D tex_HessianIJ;
uniform sampler2D tex_colormap;

#if	SHADING_BLINN_PHONG
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

TLight		uLight = TLight(
        vec4(1,1,1,0),
        vec3(.2,.2,.2),
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

	if(l_dot_n >= 0.0)
	{
		specular = light.specular * material.specular * pow(max(dot(h, n), 0.0), material.shininess);
	}
	return vec4(ambient + diffuse + specular, 1);
}



#endif




void main() {
	vec4	p = texture(tex_position, vTexCoord);
	vec3	g = texture(tex_gradient, vTexCoord).xyz;
	float	Dxy = texture(tex_HessianIJ, vTexCoord).z;
	float	tc = .01*Dxy+0.5;
	if(p.w!=0.0)
	{
#if	SHADING_BLINN_PHONG
		TMaterial	material = 
			TMaterial(
					vec3(.1,.1,.1),
					texture(tex_colormap, vec2(0,tc)).xyz,
					vec3(1,1,1),
					vec3(0,0,0),
					128.0*0.5
					);
		fColor = shade_Blinn_Phong(normalize(mat3(MV)*(-p.w*g)), MV*vec4(p.xyz,1), material, uLight);
#else
		fColor = texture(tex_colormap, vec2(0,tc));
#endif
	}
	else
		discard;
}

