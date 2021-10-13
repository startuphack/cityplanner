import logging
import logging.config
import os

import uvicorn

DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "DEBUG")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "class": "colorlog.ColoredFormatter",
            "format": (
                "%(log_color)s%(levelname)-8s%(asctime)s%(yellow)s %(name)s: %(blue)s%(message)s%(reset)s"
            ),
        },
    },
    "handlers": {
        "stderr": {
            "level": LOGGING_LEVEL,
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {
            "handlers": ["stderr"],
            "level": logging.DEBUG,
        },
    },
}

if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)
    uvicorn.run(
        "planner.app:app",
        host="0.0.0.0",
        port=8000,
        debug=DEBUG,
        reload=DEBUG,
        log_config=LOGGING,
        access_log=True,
    )
