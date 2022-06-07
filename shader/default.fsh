#version 410
#define offset 0.00001

in vec2 UV;
out vec4 FragColor;

uniform sampler2D buf;

void main()
{
	FragColor = texture(buf, UV);
}
