from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
	words: List[str]
	ori_words: List[str]
	pos_tags: List[str] = None
	synheads: List[int] = None
	syndep_labels: List[str] = None
	# semheads: List[int] =None
	# semdep_labels: List[str] = None
	labels: List[str] = None
	prediction: List[str]  = None