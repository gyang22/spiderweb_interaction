"""Entry point for Spiderweb PCD Explorer."""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtCore import QTimer
from app.main_window import MainWindow


def main() -> None:
    # Must set OpenGL format before creating QApplication
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # 4x MSAA
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("Spiderweb PCD Explorer")

    window = MainWindow()
    window.show()

    # Show the PCD selector after the event loop starts (window fully visible)
    QTimer.singleShot(0, window.show_selector_on_startup)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
