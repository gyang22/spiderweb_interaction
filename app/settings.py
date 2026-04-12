from pathlib import Path

# App-wide constants

# All file saves go here — the directory is created at startup if absent
SAVES_DIR: Path = Path(__file__).parent.parent / "saves"

SELECTION_COLOR = (1.0, 0.5, 0.0, 1.0)   # orange highlight for selected points
DEFAULT_POINT_COLOR = (0.7, 0.7, 0.7, 1.0)  # neutral gray when no RGB in file
DEFAULT_POINT_SIZE = 4.0
MIN_POINT_SIZE = 1
MAX_POINT_SIZE = 20
UNDO_MAX_DEPTH = 20
BACKGROUND_COLOR = (0.08, 0.08, 0.10, 1.0)

# How many frames between FPS label refreshes (avoids text flicker)
FPS_DISPLAY_INTERVAL = 30

# Camera defaults — adjusted at load time based on point cloud extent
CAMERA_MOVE_SPEED = 5.0       # units/second base speed
CAMERA_ACCELERATION = 40.0    # velocity ramp rate
CAMERA_DAMPING = 12.0         # friction coefficient (higher = snappier stop)
CAMERA_MOUSE_SENSITIVITY = 0.2  # degrees per pixel
CAMERA_NEAR = 0.001
CAMERA_FAR = 100000.0
CAMERA_FOV = 60.0             # vertical field of view in degrees
