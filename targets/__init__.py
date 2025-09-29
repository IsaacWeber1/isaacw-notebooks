"""
Target definitions and constraints for 38-channel LBM collision operator learning.

This module defines the learning targets and physics constraints for the simplified
38-channel approach where f and g populations are combined into unified state vectors.
"""

from .collision_targets import compute_collision_deltas, CollisionTarget
from .constraints import (
    check_mass_conservation,
    check_momentum_conservation,
    check_physical_bounds,
    validate_collision_pair
)
from .validation import validate_targets_on_data

__all__ = [
    'compute_collision_deltas',
    'CollisionTarget',
    'check_mass_conservation',
    'check_momentum_conservation',
    'check_physical_bounds',
    'validate_collision_pair',
    'validate_targets_on_data'
]