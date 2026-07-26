from __future__ import annotations

from typing import Any

from .event_aggregator import EventAggregator
from .models import EventDetection, ReviewEvent


class EvidenceEventAggregator(EventAggregator):
    """V2 amendment that records reopen operations without changing matching."""

    def observe(
        self,
        frame_number: int,
        timestamp: float,
        candidates: list[dict[str, Any]],
    ) -> list[ReviewEvent]:
        touched: list[ReviewEvent] = []
        for candidate in candidates:
            detection = EventDetection.from_candidate(
                frame_number, timestamp, candidate
            )
            event = self._match(detection, {"ACTIVE", "PENDING"})
            if event is None:
                event = self._match(detection, {"CLOSED"})
                if event is not None:
                    event.reopen_count += 1
                    event.state = "ACTIVE"
            if event is None:
                event = self._new_event(detection)
            else:
                event.add(detection)
            self._promote_if_ready(event)
            touched.append(event)
        self.advance(timestamp)
        return touched
