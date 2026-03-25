import re
from datetime import datetime
from dateutil.parser import parse
import yaml


class DateParser:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["date_parser"]

        prefixes = "|".join(cfg["prefixes"])
        self._prefix_re = re.compile(rf'^({prefixes})[:\s./]*', re.IGNORECASE)
        self.warning_days = cfg["warning_days"]

    def clean(self, raw):
        cleaned = self._prefix_re.sub('', raw).strip()
        cleaned = re.sub(r'(?<=\d)O', lambda m: '0', cleaned)
        cleaned = re.sub(r'O(?=\d)', lambda m: '0', cleaned)
        return cleaned

    def try_parse(self, raw):
        try:
            return parse(self.clean(raw))
        except (ValueError, OverflowError):
            return None

    def get_max_date(self, dates):
        if not dates:
            return None
        if isinstance(dates, str):
            dates = [dates]

        parsed = []
        for raw in dates:
            obj = self.try_parse(raw)
            if obj is not None:
                parsed.append((raw, obj))

        if not parsed:
            return dates[0]

        best_raw, _ = max(parsed, key=lambda x: x[1])
        return best_raw

    def evaluate_expiry(self, date_str):
        expiry = self.try_parse(date_str)
        if expiry is None:
            return None, None

        delta = (expiry - datetime.now()).days
        if delta < 0:
            return "expired", delta
        elif delta <= self.warning_days:
            return "warning", delta
        else:
            return "valid", delta
