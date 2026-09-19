"""MPEG-TS packet realignment when Tapo parts split 188-byte packets."""

from __future__ import annotations

TS_PACKET_SIZE = 188
SYNC = 0x47


def realign_mpegts(buffer: bytearray, incoming: bytes) -> tuple[bytes, bytearray]:
    buffer.extend(incoming)
    start = 0
    while start < len(buffer) and buffer[start] != SYNC:
        start += 1
    if start:
        del buffer[:start]
    complete = (len(buffer) // TS_PACKET_SIZE) * TS_PACKET_SIZE
    aligned = 0
    offset = 0
    while offset + TS_PACKET_SIZE <= complete:
        if buffer[offset] != SYNC:
            nxt = offset + 1
            while nxt < complete and buffer[nxt] != SYNC:
                nxt += 1
            del buffer[:nxt]
            complete = (len(buffer) // TS_PACKET_SIZE) * TS_PACKET_SIZE
            offset = 0
            continue
        aligned = offset + TS_PACKET_SIZE
        offset = aligned
    packets = bytes(buffer[:aligned])
    remaining = bytearray(buffer[aligned:])
    return packets, remaining
