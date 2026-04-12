"""
GLSL shader source strings + compile/link helpers.
All shaders target OpenGL 3.3 core profile.
"""

from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader,
    glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS, GL_LINK_STATUS, GL_TRUE,
)


# ── Main point cloud shader ────────────────────────────────────────────────────

MAIN_VERT = """
#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in float a_selected;
layout(location = 3) in float a_alive;

uniform mat4 u_mvp;
uniform float u_point_size;
uniform vec4 u_selection_color;

out vec4 v_color;

void main() {
    if (a_alive < 0.5) {
        // Move dead points outside clip space; GPU discards with zero overdraw
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        v_color = vec4(0.0);
        return;
    }

    vec4 pos = u_mvp * vec4(a_position, 1.0);
    // Pulses selected points forward by 0.002 to beat skeleton on depth
    if (a_selected > 0.5) {
        pos.z -= 0.002 * pos.w;
    }
    gl_Position = pos;
    gl_PointSize = u_point_size;
    v_color = (a_selected > 0.5) ? u_selection_color : a_color;
}
"""

MAIN_FRAG = """
#version 330 core

in vec4 v_color;
out vec4 frag_color;

void main() {
    // Circular point sprite — discard corners for round dots
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;
    frag_color = v_color;
}
"""

# ── Picking shader (color-encoded vertex ID) ───────────────────────────────────

PICK_VERT = """
#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 3) in float a_alive;

uniform mat4 u_mvp;

out vec3 v_pick_color;

void main() {
    if (a_alive < 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        v_pick_color = vec3(0.0);
        return;
    }

    // Encode (gl_VertexID + 1) as 24-bit RGB so index 0 == background (black)
    int id = gl_VertexID + 1;
    float r = float((id >> 16) & 0xFF) / 255.0;
    float g = float((id >> 8)  & 0xFF) / 255.0;
    float b = float( id        & 0xFF) / 255.0;
    v_pick_color = vec3(r, g, b);

    gl_Position = u_mvp * vec4(a_position, 1.0);
    gl_PointSize = 4.0;  // slightly enlarged for easier single-click picking
}
"""

PICK_FRAG = """
#version 330 core

in vec3 v_pick_color;
out vec4 frag_color;

void main() {
    // Full square points — no discard, preserves exact index color at every pixel
    frag_color = vec4(v_pick_color, 1.0);
}
"""


# ── Skeleton shader (nodes + edges overlay) ───────────────────────────────────

SKEL_VERT = """
#version 330 core

layout(location = 0) in vec3 a_position;

uniform mat4  u_mvp;
uniform float u_point_size;

void main() {
    vec4 pos = u_mvp * vec4(a_position, 1.0);
    // Pulls skeleton forward by 0.001 to render above unselected points, but behind selected points (which are 0.002)
    pos.z -= 0.001 * pos.w;
    gl_Position  = pos;
    gl_PointSize = u_point_size;
}
"""

SKEL_FRAG = """
#version 330 core

uniform vec4 u_color;
uniform int  u_round_points;  // 1 = circular sprite (GL_POINTS); 0 = solid (GL_LINES)
out vec4 frag_color;

void main() {
    // gl_PointCoord is undefined for GL_LINES in core profile, so skip the
    // discard test entirely when drawing lines to avoid killing all fragments.
    if (u_round_points != 0) {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (dot(coord, coord) > 0.25) discard;
    }
    frag_color = u_color;
}
"""


# ── Compile / link helpers ─────────────────────────────────────────────────────

def _compile_shader(src: str, shader_type) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation failed:\n{log}")
    return shader


def compile_program(vert_src: str, frag_src: str) -> int:
    """Compile and link a vertex+fragment program. Returns GL program ID."""
    vert = _compile_shader(vert_src, GL_VERTEX_SHADER)
    frag = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vert)
    glAttachShader(prog, frag)
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        log = glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Shader link failed:\n{log}")
    glDeleteShader(vert)
    glDeleteShader(frag)
    return prog
