#version 410

layout(location = 0) in vec4 Position;

out vec2 vTexCoord;

void main()
{
	vTexCoord = Position.xy*0.5+0.5;
	gl_Position = Position;
}
