"""Bounded camera errors. Never include credentials or protocol dumps."""


class CameraError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def unreachable() -> CameraError:
    return CameraError("camera_unreachable", "Camera is unreachable")


def auth_failed() -> CameraError:
    return CameraError("camera_auth_failed", "Camera rejected the credentials")


def unsupported() -> CameraError:
    return CameraError("camera_stream_unsupported", "Camera did not return a usable stream")


def timed_out() -> CameraError:
    return CameraError("camera_stream_timeout", "Camera stream timed out")


def failed() -> CameraError:
    return CameraError("camera_stream_failed", "Camera stream failed")
