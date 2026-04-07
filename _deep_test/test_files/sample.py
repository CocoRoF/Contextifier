"""Sample Python module for testing."""

import os
import sys


class DataProcessor:
    """Process data from various sources."""

    def __init__(self, config: dict):
        self.config = config
        self._cache = {}

    def process(self, data: list) -> list:
        """Process a list of data items."""
        results = []
        for item in data:
            if item not in self._cache:
                self._cache[item] = self._transform(item)
            results.append(self._cache[item])
        return results

    def _transform(self, item):
        return item.upper() if isinstance(item, str) else item
