"""Simple UDP bridge stub for communicating predictions to Unity."""

import json
import socket
from typing import Any


def send_prediction_udp(payload: dict[str, Any], host: str = "127.0.0.1", port: int = 5005) -> None:
    """Send one prediction payload to Unity via UDP.

    TODO: Replace with finalized transport/protocol once Unity bridge spec is fixed.
    """
    message = json.dumps(payload).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message, (host, port))
    finally:
        sock.close()
