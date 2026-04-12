"""
SkeletonRenderer — renders a StrandGraph as GL_LINES (edges) + GL_POINTS (nodes).

Must be initialized inside initializeGL() after the GL context is current.

Lifecycle::

    renderer = SkeletonRenderer()
    renderer.initialize()       # compile shaders — call once in initializeGL()
    renderer.upload(graph)      # upload graph data — call with context current
    renderer.draw(mvp)          # call from paintGL()
    renderer.clear()            # hide overlay (no GPU free)
"""

from __future__ import annotations
import numpy as np

from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glDeleteVertexArrays,
    glGenBuffers, glBindBuffer, glBufferData, glDeleteBuffers,
    glVertexAttribPointer, glEnableVertexAttribArray,
    glUseProgram, glUniformMatrix4fv, glUniform4f, glUniform1f, glUniform1i,
    glGetUniformLocation,
    glDrawArrays, glDrawElements,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER,
    GL_STATIC_DRAW, GL_FLOAT, GL_FALSE,
    GL_POINTS, GL_LINES, GL_UNSIGNED_INT,
)

from app.gl.shaders import compile_program, SKEL_VERT, SKEL_FRAG
from app.data.strand_graph import StrandGraph


class SkeletonRenderer:
    def __init__(self) -> None:
        self._program: int | None = None

        # Edge VAO: position VBO + element buffer for GL_LINES
        self._vao_edges: int | None = None
        self._vbo_edge_pos: int | None = None
        self._ebo_edges: int | None = None

        # Node VAO: separate position VBO for GL_POINTS
        self._vao_nodes: int | None = None
        self._vbo_node_pos: int | None = None

        self._n_nodes: int = 0
        self._n_edge_indices: int = 0   # 2 * n_edges for GL_LINES

        # Visual settings
        self.node_size:  float = 8.0
        self.line_width: float = 2.0
        self.node_color: tuple = (1.0, 0.8, 0.0, 1.0)    # gold
        self.edge_color: tuple = (0.6, 0.9, 1.0, 0.85)   # light blue

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Compile shaders. Call inside initializeGL()."""
        self._program = compile_program(SKEL_VERT, SKEL_FRAG)

    def upload(self, graph: StrandGraph) -> None:
        """
        Upload graph geometry to the GPU, replacing any previous data.
        Call with the GL context current (use viewport.makeCurrent() beforehand).
        """
        self._free_gpu()
        if len(graph.nodes) == 0:
            return

        nodes = np.ascontiguousarray(graph.nodes, dtype=np.float32)  # (M, 3)
        edges = np.ascontiguousarray(graph.edges, dtype=np.uint32)   # (E, 2)

        self._n_nodes = len(nodes)
        self._n_edge_indices = edges.size   # 2 * E

        # ── Edge VAO ──────────────────────────────────────────────────────────
        self._vao_edges = glGenVertexArrays(1)
        self._vbo_edge_pos, self._ebo_edges = glGenBuffers(2)

        glBindVertexArray(self._vao_edges)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_edge_pos)
        glBufferData(GL_ARRAY_BUFFER, nodes.nbytes, nodes, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo_edges)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, edges.nbytes, edges, GL_STATIC_DRAW)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # ── Node VAO (separate VBO so we can draw GL_POINTS without EBO) ──────
        self._vao_nodes = glGenVertexArrays(1)
        self._vbo_node_pos = glGenBuffers(1)

        glBindVertexArray(self._vao_nodes)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_node_pos)
        glBufferData(GL_ARRAY_BUFFER, nodes.nbytes, nodes, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, mvp: np.ndarray) -> None:
        """Render edges then nodes. Call from paintGL()."""
        if self._program is None or self._n_nodes == 0:
            return

        glUseProgram(self._program)

        loc_mvp    = glGetUniformLocation(self._program, 'u_mvp')
        loc_color  = glGetUniformLocation(self._program, 'u_color')
        loc_size   = glGetUniformLocation(self._program, 'u_point_size')
        loc_round  = glGetUniformLocation(self._program, 'u_round_points')

        glUniformMatrix4fv(loc_mvp, 1, True, mvp)

        # Draw edges (GL_LINES) — u_round_points=0 so the discard is skipped
        if self._n_edge_indices > 0:
            glUniform4f(loc_color, *self.edge_color)
            glUniform1f(loc_size, 1.0)
            glUniform1i(loc_round, 0)
            try:
                from OpenGL.GL import glLineWidth
                glLineWidth(self.line_width)
            except Exception:
                pass
            glBindVertexArray(self._vao_edges)
            glDrawElements(GL_LINES, self._n_edge_indices, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

        # Draw nodes (GL_POINTS) — u_round_points=1 enables circular sprite discard
        glUniform4f(loc_color, *self.node_color)
        glUniform1f(loc_size, self.node_size)
        glUniform1i(loc_round, 1)
        glBindVertexArray(self._vao_nodes)
        glDrawArrays(GL_POINTS, 0, self._n_nodes)
        glBindVertexArray(0)

    def clear(self) -> None:
        """Hide the skeleton overlay (zeroes draw counts; GPU buffers remain)."""
        self._n_nodes = 0
        self._n_edge_indices = 0

    @property
    def has_data(self) -> bool:
        return self._n_nodes > 0

    # ── GPU resource management ───────────────────────────────────────────────

    def _free_gpu(self) -> None:
        if self._vao_edges is not None:
            glDeleteVertexArrays(1, [self._vao_edges])
            self._vao_edges = None
        if self._vbo_edge_pos is not None:
            glDeleteBuffers(1, [self._vbo_edge_pos])
            self._vbo_edge_pos = None
        if self._ebo_edges is not None:
            glDeleteBuffers(1, [self._ebo_edges])
            self._ebo_edges = None
        if self._vao_nodes is not None:
            glDeleteVertexArrays(1, [self._vao_nodes])
            self._vao_nodes = None
        if self._vbo_node_pos is not None:
            glDeleteBuffers(1, [self._vbo_node_pos])
            self._vbo_node_pos = None
        self._n_nodes = 0
        self._n_edge_indices = 0
