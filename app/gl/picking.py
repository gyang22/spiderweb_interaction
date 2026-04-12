"""
PickingRenderer — offscreen FBO with color-encoded vertex IDs for GPU picking.

Index encoding:
  Background  →  RGB(0, 0, 0)
  Point i     →  RGB encoding of (i + 1) as 24-bit integer
So decode: index = (R<<16 | G<<8 | B) - 1;  -1 means background.
"""

import numpy as np
from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glFramebufferTexture2D,
    glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage,
    glFramebufferRenderbuffer,
    glCheckFramebufferStatus,
    glDeleteFramebuffers, glDeleteTextures, glDeleteRenderbuffers,
    glViewport,
    glUseProgram, glUniformMatrix4fv, glGetUniformLocation,
    glClearColor, glClear, glDisable, glEnable, glReadPixels,
    glBindVertexArray, glDrawArrays,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST,
    GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
    GL_FRAMEBUFFER_COMPLETE,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_MULTISAMPLE, GL_POINTS,
)

from app.gl.shaders import compile_program, PICK_VERT, PICK_FRAG


class PickingRenderer:
    def __init__(self):
        self._program = None
        self._fbo = None
        self._color_tex = None
        self._depth_rb = None
        self._width = 1
        self._height = 1

    # ── init (call inside initializeGL) ──────────────────────────────────────

    def initialize(self) -> None:
        self._program = compile_program(PICK_VERT, PICK_FRAG)

    def _create_fbo(self, w: int, h: int) -> None:
        """(Re)create the offscreen FBO at the given size."""
        # Clean up existing resources
        if self._fbo is not None:
            glDeleteFramebuffers(1, [self._fbo])
            glDeleteTextures(1, [self._color_tex])
            glDeleteRenderbuffers(1, [self._depth_rb])

        self._width = max(1, w)
        self._height = max(1, h)

        # Color texture — GL_RGB8 for exact integer readback
        self._color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self._width, self._height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Depth renderbuffer
        self._depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self._width, self._height)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        # Assemble FBO
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self._color_tex, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, self._depth_rb)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Picking FBO incomplete: status=0x{status:X}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def resize(self, w: int, h: int) -> None:
        self._create_fbo(w, h)

    @property
    def fbo_width(self) -> int:
        return self._width

    @property
    def fbo_height(self) -> int:
        return self._height

    # ── picking render pass ───────────────────────────────────────────────────

    def render(self, renderer, mvp: np.ndarray) -> None:
        """Render all points into the picking FBO with index-encoded colors."""
        if self._fbo is None:
            self._create_fbo(self._width, self._height)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        # QPainter (called from paintEvent overlay) modifies glViewport and leaves
        # it in an indeterminate state.  Reset it to match the FBO size so the
        # picking render maps to the same screen region as the main render.
        glViewport(0, 0, self._width, self._height)
        try:
            glDisable(GL_MULTISAMPLE)  # AA would corrupt encoded colors
        except Exception:
            pass

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._program)
        loc = glGetUniformLocation(self._program, 'u_mvp')
        glUniformMatrix4fv(loc, 1, True, mvp)

        glBindVertexArray(renderer.vao)
        glDrawArrays(GL_POINTS, 0, renderer.n_points)
        glBindVertexArray(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        try:
            glEnable(GL_MULTISAMPLE)
        except Exception:
            pass

    # ── pixel readback ────────────────────────────────────────────────────────

    def read_pixel(self, x: int, y: int) -> int:
        """Read single pixel from picking FBO. Returns point index (0-based) or -1."""
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        pixel = glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        pixel = np.frombuffer(pixel, dtype=np.uint8)
        r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
        encoded = (r << 16) | (g << 8) | b
        return encoded - 1  # -1 means background

    def read_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read a rectangle from the picking FBO. Returns 1D array of unique alive indices."""
        if w <= 0 or h <= 0:
            return np.array([], dtype=np.int32)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        pixels = glReadPixels(x, y, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3)
        return self._decode_pixels(arr)

    def read_full(self) -> np.ndarray:
        """Read the entire picking FBO. Returns 2D array [H, W] of indices (-1=background)."""
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        pixels = glReadPixels(0, 0, self._width, self._height, GL_RGB, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(self._height, self._width, 3)
        r = arr[:, :, 0].astype(np.int32)
        g = arr[:, :, 1].astype(np.int32)
        b = arr[:, :, 2].astype(np.int32)
        encoded = (r << 16) | (g << 8) | b  # 0 = background
        return encoded - 1  # background becomes -1

    @staticmethod
    def _decode_pixels(arr: np.ndarray) -> np.ndarray:
        """Decode RGB pixel array → unique point indices (excluding background -1)."""
        r = arr[:, :, 0].astype(np.int32)
        g = arr[:, :, 1].astype(np.int32)
        b = arr[:, :, 2].astype(np.int32)
        encoded = (r << 16) | (g << 8) | b
        unique = np.unique(encoded)
        indices = unique[unique > 0] - 1  # remove background (0), convert to 0-based
        return indices.astype(np.int32)
