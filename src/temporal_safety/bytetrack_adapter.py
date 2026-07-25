from __future__ import annotations

from .tracker_base import CausalTracker


class ByteTrackAdapter(CausalTracker):
    """Auditable two-stage high/low-confidence ByteTrack adapter."""

    name = "bytetrack"

