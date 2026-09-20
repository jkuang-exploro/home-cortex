"""Leaf-level constants and errors shared by spatial domain modules."""

LENGTH_UNIT = "m"
ANGLE_UNIT = "rad"
TIME_UNIT = "s"
POSE_FIELDS = frozenset({"orientation", "position"})


class SpatialContractError(ValueError):
    """A stored or runtime spatial value violates its declared contract."""
