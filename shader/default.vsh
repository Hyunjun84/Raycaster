#version 410

layout(location = 0) in vec4 Position;

out vec2 UV;

void main()
{
	UV = (Position.xy+1)*0.5;
	gl_Position = Position;
}
