from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "tickyantra.app:app",
        host=os.getenv("TICKYANTRA_HOST", "0.0.0.0"),
        port=int(os.getenv("TICKYANTRA_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
