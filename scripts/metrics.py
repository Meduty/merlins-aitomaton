"""
Statistics and metrics tracking module for MTG Card Generator.
Handles card generation statistics without global variables.
"""

import threading
import time
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class GenerationMetrics:
    """Container for generation metrics with thread-safe access."""
    
    colors: Dict[str, int] = field(default_factory=dict)
    rarities: Dict[str, int] = field(default_factory=dict)
    successful: int = 0
    total_runtime: float = 0.0
    all_cards: list = field(default_factory=list)
    
    # Thread locks
    _color_lock: threading.Lock = field(default_factory=threading.Lock)
    _rarity_lock: threading.Lock = field(default_factory=threading.Lock)
    _successful_lock: threading.Lock = field(default_factory=threading.Lock)
    _runtime_lock: threading.Lock = field(default_factory=threading.Lock)
    _cards_lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update_color(self, color_identity: str) -> None:
        """Thread-safe color count update."""
        with self._color_lock:
            color_key = color_identity.lower()
            if color_key in self.colors:
                self.colors[color_key] += 1
            else:
                self.colors[color_key] = 1
    
    def update_rarity(self, rarity: str) -> None:
        """Thread-safe rarity count update."""
        with self._rarity_lock:
            rarity_key = rarity.lower()
            if rarity_key in self.rarities:
                self.rarities[rarity_key] += 1
            else:
                self.rarities[rarity_key] = 1
    
    def increment_successful(self) -> None:
        """Thread-safe successful count increment."""
        with self._successful_lock:
            self.successful += 1
    
    def add_runtime(self, runtime: float) -> None:
        """Thread-safe runtime addition."""
        with self._runtime_lock:
            self.total_runtime += runtime
    
    def add_card(self, card_data: Dict[str, Any]) -> None:
        """Thread-safe card addition."""
        with self._cards_lock:
            self.all_cards.append(card_data)
    
    def get_average_time_per_card(self) -> float:
        """Calculate average time per card."""
        with self._successful_lock, self._runtime_lock:
            return self.total_runtime / self.successful if self.successful > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with (self._color_lock, self._rarity_lock, 
              self._successful_lock, self._runtime_lock):
            return {
                "colors": dict(self.colors),
                "rarities": dict(self.rarities),
                "successful": self.successful,
                "total_runtime": self.total_runtime,
                "average_time_per_card": self.get_average_time_per_card(),
                "total_cards": len(self.all_cards)
            }
