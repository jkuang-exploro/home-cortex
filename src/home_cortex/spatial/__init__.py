"""Graph spatial contracts and pure geometry.

Import concrete modules explicitly. Keeping this package initializer empty avoids
loading optional anchor-observation and localization math during chat startup.
Device polling, detectors, SLAM, and localization loops belong in
``home_cortex_client``. Canonical fused telemetry is accepted by
``spatial.presence``; this initializer does not load that module.
"""
