import shelve
import os
from config.settings import settings

class QCache:
    def __init__(self):
        os.makedirs(os.path.dirname(settings.CACHE_PATH), exist_ok=True)
        self.cache_file = settings.CACHE_PATH

    def get(self, question: str):
        """Retrieve answer if exists."""
        clean_q = question.strip().lower()
        with shelve.open(self.cache_file) as db:
            if clean_q in db:
                return db[clean_q]
        return None

    def set(self, question: str, answer: str):
        """Store validated answer."""
        clean_q = question.strip().lower()
        with shelve.open(self.cache_file) as db:
            db[clean_q] = answer