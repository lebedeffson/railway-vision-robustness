from __future__ import annotations

import json

from canonical_m4_common import create_protocol_lock


if __name__ == "__main__":
    print(json.dumps(create_protocol_lock(), indent=2, ensure_ascii=False))
