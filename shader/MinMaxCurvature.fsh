#version 330 core

#define	SHADING_BLINN_PHONG	1

in vec2 vTexCoord;

out vec4 fColor;

uniform mat4        MV;

uniform sampler2D tex_position;
uniform sampler2D tex_gradient;
uniform sampler2D tex_HessianII;
uniform sampler2D tex_HessianIJ;
uniform sampler2D tex_colormap;
uniform int orientation;

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

	if(l_dot_n >= 0.0)
	{
		specular = light.specular * material.specular * pow(max(dot(h, n), 0.0), material.shininess);
	}
	return vec4(ambient + diffuse + specular, 1);
}

#endif

void main() {
	vec4	p = texture(tex_position, vTexCoord);
	vec3	g = -texture(tex_gradient, vTexCoord).xyz;
	vec3	dii = texture(tex_HessianII, vTexCoord).xyz;
	vec3	dij = texture(tex_HessianIJ, vTexCoord).xyz;

	mat3	H = mat3(dii.x, dij.z, dij.y,
			 dij.z, dii.y, dij.x,
                         dij.y, dij.x, dii.z)*orientation;

	float	one_over_len_g = 1.0/length(g);
	vec3	n = g*one_over_len_g;
	mat3    P = mat3(1.0) - mat3(n.x*n.x, n.x*n.y, n.x*n.z,
				     n.x*n.y, n.y*n.y, n.y*n.z,
                                     n.x*n.z, n.y*n.z, n.z*n.z);
	mat3    M = -P*H*P*one_over_len_g;
	float   T = M[0][0] + M[1][1] + M[2][2];
	mat3    MMt = M*transpose(M);
	float   F = sqrt(MMt[0][0] + MMt[1][1] + MMt[2][2]);
	float   k_max = (T + sqrt(2.0*F*F - T*T))*0.5;
	float   k_min = (T - sqrt(2.0*F*F - T*T))*0.5;

	float	scale_k = 10;
	vec2	tc = vec2(scale_k*vec2(k_max,k_min)+0.5);

	if(p.w!=0.0)
	{
#if	SHADING_BLINN_PHONG
		TMaterial	material = 
			TMaterial(
				vec3(.2,.2,.2),
				texture(tex_colormap, tc).xyz,
				vec3(1,1,1),
				vec3(0,0,0),
				128.0*0.5
				);
		fColor = shade_Blinn_Phong(normalize(mat3(MV)*(p.w*g)), MV*vec4(p.xyz,1), material, uLight);
#else
		fColor = texture(tex_colormap, tc);
#endif
	}
	else
		discard;
}

