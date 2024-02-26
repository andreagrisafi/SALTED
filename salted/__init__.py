import logging

logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_log = logging.getLogger(__name__)
