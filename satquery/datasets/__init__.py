"""Dataset adapters.

Adapters are reached through the registry (`satquery.registry.build_dataset`)
rather than imported from here, so a broken or missing adapter can never take
down `satquery datasets` for the other nine.
"""

from .base import BaseDataset, first_existing, read_csv_dicts, read_json, read_jsonl

__all__ = ["BaseDataset", "first_existing", "read_csv_dicts", "read_json", "read_jsonl"]
