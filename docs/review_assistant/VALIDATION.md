# MVP validation

All validation used development/train or synthetic inputs. Railway test remained
sealed with access count zero.

## Long throughput benchmark

Input: 1,100-frame local video constructed by repeating one 100-frame
development-train sequence. This is a throughput workload, not independent
accuracy evidence.

```text
warm-up frames:          20
measured frames:         1000
end-to-end FPS:          15.69
detector FPS:            36.48
verifier FPS:            33.12
event aggregation:       0.204 ms/frame
video encoding:          4.876 ms/frame
peak GPU memory:         467,382,784 bytes
peak RAM:                1,504,337,920 bytes
real-time claim:         false
offline processing:      supported
```

## Frozen-pipeline product smoke

A 100-frame development-train clip was processed through the frozen detector,
OC-SORT, combined verifier, event aggregator, clip writer, SQLite database and
HTML/PDF report.

```text
frames:                  100
raw detections:          296
unique review events:    3
end-to-end FPS:          16.94
report/database parity:  PASS
autonomous alarms:       0
safety actuations:       0
```

The reduction from frame detections to event cards demonstrates interaction
compression on this smoke input. It is not a false-positive-rate or safety
performance claim.
