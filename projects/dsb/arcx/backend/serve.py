"""Launch the ArcX API server"""

import logging
import sys

import uvicorn

from arcx.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """Run the API server"""
    logger.info("Starting ArcX API server...")
    logger.info(f"Host: {config.api.host}:{config.api.port}")

    uvicorn.run(
        "arcx.api.server:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
