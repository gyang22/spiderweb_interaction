"""
PointCloudRenderer — VAO/VBO management and draw calls.

VBO layout (4 buffers, all GL_DYNAMIC_DRAW):
  loc 0  positions   float32 (N, 3)   — fixed after load
  loc 1  colors      float32 (N, 4)   — dirty on color edit
  loc 2  selection   float32 (N,)     — dirty on selection change
  loc 3  alive       float32 (N,)     — dirty on delete/undo
"""

import ctypes
import numpy as np

from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray,
    glGenBuffers, glBindBuffer, glBufferData, glBufferSubData,
    glVertexAttribPointer, glEnableVertexAttribArray,
    glUseProgram, glUniformMatrix4fv, glUniform1f, glUniform4f,
    glGetUniformLocation,
    glDrawArrays,
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE, GL_POINTS,
)

from app.gl.shaders import compile_program, MAIN_VERT, MAIN_FRAG
from app import settings


class PointCloudRenderer:
    def __init__(self):
        self._program = None
        self._vao = None
        self._vbo_positions = None
        self._vbo_colors = None
        self._vbo_selection = None
        self._vbo_alive = None

        self._pc = None          # reference to active PointCloud
        self._n = 0              # total allocated point count

        self.point_size = settings.DEFAULT_POINT_SIZE

        # Dirty flags — True means the VBO needs to be re-uploaded
        self._colors_dirty = False
        self._selection_dirty = False
        self._alive_dirty = False

    # ── init (call inside initializeGL) ──────────────────────────────────────

    def initialize(self) -> None:
        self._program = compile_program(MAIN_VERT, MAIN_FRAG)

    # ── load point cloud ──────────────────────────────────────────────────────

    def load(self, pc) -> None:
        """Upload point cloud data to GPU. Allocates new VBOs."""
        self._pc = pc
        self._n = pc.total_count

        if self._vao is None:
            self._vao = glGenVertexArrays(1)
        if self._vbo_positions is None:
            vbos = glGenBuffers(4)
            self._vbo_positions, self._vbo_colors, self._vbo_selection, self._vbo_alive = vbos

        glBindVertexArray(self._vao)

        # Positions (loc 0)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, pc.positions.nbytes, pc.positions, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Colors (loc 1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, pc.colors.nbytes, pc.colors, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Selection (loc 2)
        sel = pc.selection_float()
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_selection)
        glBufferData(GL_ARRAY_BUFFER, sel.nbytes, sel, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        # Alive (loc 3)
        alive = pc.alive_float()
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_alive)
        glBufferData(GL_ARRAY_BUFFER, alive.nbytes, alive, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(3)

        glBindVertexArray(0)

        self._colors_dirty = False
        self._selection_dirty = False
        self._alive_dirty = False

    # ── dirty markers (call after modifying pc data) ──────────────────────────

    def mark_colors_dirty(self):    self._colors_dirty = True
    def mark_selection_dirty(self): self._selection_dirty = True
    def mark_alive_dirty(self):     self._alive_dirty = True

    def upload_dirty_vbos(self) -> None:
        if self._pc is None:
            return
        if self._colors_dirty:
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_colors)
            glBufferSubData(GL_ARRAY_BUFFER, 0, self._pc.colors.nbytes, self._pc.colors)
            self._colors_dirty = False
        if self._selection_dirty:
            sel = self._pc.selection_float()
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_selection)
            glBufferSubData(GL_ARRAY_BUFFER, 0, sel.nbytes, sel)
            self._selection_dirty = False
        if self._alive_dirty:
            alive = self._pc.alive_float()
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_alive)
            glBufferSubData(GL_ARRAY_BUFFER, 0, alive.nbytes, alive)
            self._alive_dirty = False

    # ── draw ─────────────────────────────────────────────────────────────────

    def draw(self, mvp: np.ndarray) -> None:
        if self._pc is None or self._n == 0:
            return

        self.upload_dirty_vbos()

        glUseProgram(self._program)

        loc_mvp = glGetUniformLocation(self._program, 'u_mvp')
        glUniformMatrix4fv(loc_mvp, 1, True, mvp)

        loc_ps = glGetUniformLocation(self._program, 'u_point_size')
        glUniform1f(loc_ps, float(self.point_size))

        sc = settings.SELECTION_COLOR
        loc_sc = glGetUniformLocation(self._program, 'u_selection_color')
        glUniform4f(loc_sc, *sc)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self._n)
        glBindVertexArray(0)

    def clear(self) -> None:
        """Detach the point cloud so nothing is drawn (no GPU resources freed)."""
        self._pc = None
        self._n = 0

    # ── accessors needed by PickingRenderer ───────────────────────────────────

    @property
    def has_data(self) -> bool:
        return self._pc is not None and self._n > 0

    @property
    def vao(self):
        return self._vao

    @property
    def n_points(self) -> int:
        return self._n
