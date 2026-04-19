"""
SkeletonRenderer — renders a StrandGraph as GL_LINES (edges) + GL_POINTS (nodes).

Supports per-node selection highlight via upload_selection().
"""

from __future__ import annotations
import numpy as np

from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glDeleteVertexArrays,
    glGenBuffers, glBindBuffer, glBufferData, glBufferSubData, glDeleteBuffers,
    glVertexAttribPointer, glEnableVertexAttribArray,
    glUseProgram, glUniformMatrix4fv, glUniform4f, glUniform1f, glUniform1i,
    glGetUniformLocation,
    glDrawArrays, glDrawElements,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER,
    GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE,
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

        # Node VAO: position VBO + selection flag VBO for GL_POINTS
        self._vao_nodes: int | None = None
        self._vbo_node_pos: int | None = None
        self._vbo_node_sel: int | None = None   # per-node float selection flag

        self._n_nodes: int = 0
        self._n_edge_indices: int = 0

        # Visual settings
        self.node_size:      float = 8.0
        self.line_width:     float = 2.0
        self.node_color:     tuple = (1.0, 0.8, 0.0, 1.0)    # gold
        self.edge_color:     tuple = (0.6, 0.9, 1.0, 0.85)   # light blue
        self.selected_color: tuple = (1.0, 0.15, 0.15, 1.0)   # red

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        self._program = compile_program(SKEL_VERT, SKEL_FRAG)

    def upload(self, graph: StrandGraph) -> None:
        self._free_gpu()
        if len(graph.nodes) == 0:
            return

        nodes = np.ascontiguousarray(graph.nodes, dtype=np.float32)
        edges = np.ascontiguousarray(graph.edges, dtype=np.uint32)

        self._n_nodes = len(nodes)
        self._n_edge_indices = edges.size

        # ── Edge VAO ──────────────────────────────────────────────────────────
        self._vao_edges = glGenVertexArrays(1)
        self._vbo_edge_pos, self._ebo_edges = glGenBuffers(2)

        glBindVertexArray(self._vao_edges)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_edge_pos)
        glBufferData(GL_ARRAY_BUFFER, nodes.nbytes, nodes, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        # location 1 (a_selected) deliberately NOT enabled → defaults to 0.0
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo_edges)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, edges.nbytes, edges, GL_STATIC_DRAW)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # ── Node VAO ──────────────────────────────────────────────────────────
        self._vao_nodes = glGenVertexArrays(1)
        self._vbo_node_pos, self._vbo_node_sel = glGenBuffers(2)

        sel_data = np.zeros(self._n_nodes, dtype=np.float32)

        glBindVertexArray(self._vao_nodes)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_node_pos)
        glBufferData(GL_ARRAY_BUFFER, nodes.nbytes, nodes, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_node_sel)
        glBufferData(GL_ARRAY_BUFFER, sel_data.nbytes, sel_data, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def upload_selection(self, mask: np.ndarray) -> None:
        """Update per-node selection highlight. mask is bool array length n_nodes."""
        if self._vbo_node_sel is None or self._n_nodes == 0:
            return
        data = np.ascontiguousarray(mask, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_node_sel)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, mvp: np.ndarray) -> None:
        if self._program is None or self._n_nodes == 0:
            return

        from OpenGL.GL import (
            glEnable, glDisable, glDepthFunc, glGetIntegerv,
            GL_DEPTH_FUNC, GL_LEQUAL, GL_BLEND, glBlendFunc,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        )

        glUseProgram(self._program)

        loc_mvp      = glGetUniformLocation(self._program, 'u_mvp')
        loc_color    = glGetUniformLocation(self._program, 'u_color')
        loc_sel_col  = glGetUniformLocation(self._program, 'u_selected_color')
        loc_size     = glGetUniformLocation(self._program, 'u_point_size')
        loc_round    = glGetUniformLocation(self._program, 'u_round_points')

        glUniformMatrix4fv(loc_mvp, 1, True, mvp)
        glUniform4f(loc_sel_col, *self.selected_color)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        old_depth_func = glGetIntegerv(GL_DEPTH_FUNC)
        glDepthFunc(GL_LEQUAL)

        # Edges
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

        # Nodes
        glUniform4f(loc_color, *self.node_color)
        glUniform1f(loc_size, self.node_size)
        glUniform1i(loc_round, 1)
        glBindVertexArray(self._vao_nodes)
        glDrawArrays(GL_POINTS, 0, self._n_nodes)
        glBindVertexArray(0)

        glDepthFunc(old_depth_func)
        glDisable(GL_BLEND)

    def clear(self) -> None:
        self._n_nodes = 0
        self._n_edge_indices = 0

    @property
    def has_data(self) -> bool:
        return self._n_nodes > 0

    # ── GPU resource management ───────────────────────────────────────────────

    def _free_gpu(self) -> None:
        if self._vao_edges is not None:
            glDeleteVertexArrays(1, [self._vao_edges]); self._vao_edges = None
        if self._vbo_edge_pos is not None:
            glDeleteBuffers(1, [self._vbo_edge_pos]); self._vbo_edge_pos = None
        if self._ebo_edges is not None:
            glDeleteBuffers(1, [self._ebo_edges]); self._ebo_edges = None
        if self._vao_nodes is not None:
            glDeleteVertexArrays(1, [self._vao_nodes]); self._vao_nodes = None
        if self._vbo_node_pos is not None:
            glDeleteBuffers(1, [self._vbo_node_pos]); self._vbo_node_pos = None
        if self._vbo_node_sel is not None:
            glDeleteBuffers(1, [self._vbo_node_sel]); self._vbo_node_sel = None
        self._n_nodes = 0
        self._n_edge_indices = 0
