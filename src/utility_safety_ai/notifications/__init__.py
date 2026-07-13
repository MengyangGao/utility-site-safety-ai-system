"""Optional outbound notification adapters."""

from .webhook import deliver_webhook

__all__ = ["deliver_webhook"]
