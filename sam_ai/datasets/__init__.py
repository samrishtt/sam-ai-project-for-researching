# SAM-AI Datasets Package
from sam_ai.datasets.logic_tasks import get_logic_tasks
from sam_ai.datasets.math_reasoning import get_math_tasks
from sam_ai.datasets.pattern_tasks import get_pattern_tasks
from sam_ai.datasets.adversarial_tasks import get_adversarial_tasks, get_adversarial_tasks_by_type

__all__ = [
    "get_logic_tasks",
    "get_math_tasks",
    "get_pattern_tasks",
    "get_adversarial_tasks",
    "get_adversarial_tasks_by_type",
]
