#version 450 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec4 aColor;

out vec4 particleColor;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * vec4(aPos.xyz, 1.0);
    particleColor = aColor;
    gl_PointSize = 2.0;
} 