"""Development CLI: python -m home_cortex.vision.edge"""
from __future__ import annotations

import argparse
import sys

from .frames import DEV_CAMERA_ID, DEV_DEVICE_ID
from .runtime import EdgeRuntime
from .sources import MacCameraSource, SyntheticCameraSource
from .stream import StreamConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EdgeVision live camera runtime (not the Home Cortex API)."
    )
    parser.add_argument(
        "--source",
        choices=("mac", "synthetic"),
        default="mac",
        help="mac uses the built-in camera; synthetic needs no hardware",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="listen address (default: all interfaces for the Cortex server; use 127.0.0.1 for local-only)",
    )
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device-id", default=DEV_DEVICE_ID)
    parser.add_argument("--camera-id", default=DEV_CAMERA_ID)
    parser.add_argument("--index", type=int, default=0, help="OpenCV camera index")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = StreamConfig(host=args.host, port=args.port)
    if args.source == "synthetic":
        source = SyntheticCameraSource(
            device_id=args.device_id,
            camera_id=args.camera_id,
            width=args.width or 640,
            height=args.height or 480,
            fps=args.fps,
        )
    else:
        source = MacCameraSource(
            index=args.index,
            device_id=args.device_id,
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    runtime = EdgeRuntime(source, config=config, fps=args.fps)
    endpoint = runtime.start()
    print("EdgeVision")
    print(f"  device_id={source.device_id}")
    print(f"  camera_id={source.camera_id}")
    print(f"  source={args.source}")
    print(f"  transport={config.transport}")
    print(f"  stream={endpoint}")
    print(f"  viewer={config.viewer}")
    if args.host == "0.0.0.0":
        print("  LAN access: use this Mac's LAN IP in VISION_STREAM_URL, not 0.0.0.0")
    print("  stop=Ctrl-C")
    try:
        while runtime.running:
            runtime.wait(0.5)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
