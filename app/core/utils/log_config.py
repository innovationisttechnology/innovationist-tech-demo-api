import logging

_GREEN = "\x1b[32m"
_CYAN = "\x1b[36m"
_RESET = "\x1b[0m"


class _JobFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        parts = record.name.split(".")
        record.name = parts[1].capitalize() if len(parts) > 1 else parts[0].capitalize()
        record.levelname = f"{_GREEN}{record.levelname}{_RESET}"
        formatted = super().format(record)
        prefix, _, message = formatted.partition(" - ")
        return f"{prefix} - {_CYAN}{message}{_RESET}"


def init_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JobFormatter("%(levelname)s:     %(name)s - %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
