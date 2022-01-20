from typing import List, Dict, Tuple
from instance import Instance
from termcolor import colored

B_PREF="B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
O = "O"

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"
ROOT = "<ROOT>"
unk_id = -1
root_dep_label = "root"
self_label = "self"

print(colored("[Info] remember to chec the root dependency label if changing the data. current: {}".format(root_dep_label), "red"  ))

def convert_iobes(labels: List[str]) -> List[str]:
	"""
	Use IOBES tagging schema to replace the IOB tagging schema in the instance
	:param insts:
	:return:
	"""
	for pos in range(len(labels)):
		curr_entity = labels[pos]
		if pos == len(labels) - 1:
			if curr_entity.startswith(B_PREF):
				labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				labels[pos] = curr_entity.replace(I_PREF, E_PREF)
		else:
			next_entity = labels[pos + 1]
			if curr_entity.startswith(B_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(I_PREF, E_PREF)
	return labels


def build_label_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
	"""
	Build the mapping from label to index and index to labels.
	:param insts: list of instances.
	:return:
	"""
	label2idx = {}
	idx2labels = []
	label2idx[PAD] = len(label2idx)
	idx2labels.append(PAD)
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				idx2labels.append(label)
				label2idx[label] = len(label2idx)

	label2idx[START_TAG] = len(label2idx)
	idx2labels.append(START_TAG)
	label2idx[STOP_TAG] = len(label2idx)
	idx2labels.append(STOP_TAG)
	label_size = len(label2idx)
	print("#labels: {}".format(label_size))
	print("label 2idx: {}".format(label2idx))
	return idx2labels, label2idx

def check_all_labels_in_dict(insts: List[Instance], label2idx: Dict[str, int]):
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				raise ValueError(f"The label {label} does not exist in label2idx dict. The label might not appear in the training set.")


def build_word_idx(trains:List[Instance], devs:List[Instance], tests:List[Instance]) -> Tuple[Dict, List, Dict, List]:
	"""
	Build the vocab 2 idx for all instances
	:param train_insts:
	:param dev_insts:
	:param test_insts:
	:return:
	"""
	word2idx = dict()
	idx2word = []
	word2idx[PAD] = 0
	idx2word.append(PAD)
	word2idx[UNK] = 1
	idx2word.append(UNK)

	char2idx = {}
	idx2char = []
	char2idx[PAD] = 0
	idx2char.append(PAD)
	char2idx[UNK] = 1
	idx2char.append(UNK)

	# extract char on train, dev, test
	for inst in trains + devs + tests:
		for word in inst.words:
			if word not in word2idx:
				word2idx[word] = len(word2idx)
				idx2word.append(word)
	# extract char only on train (doesn't matter for dev and test)
	for inst in trains:
		for word in inst.words:
			for c in word:
				if c not in char2idx:
					char2idx[c] = len(idx2char)
					idx2char.append(c)
	return word2idx, idx2word, char2idx, idx2char


def check_all_obj_is_None(objs):
	for obj in objs:
		if obj is not None:
			return False
	return [None] * len(objs)


def build_deplabel_idx(insts: List[Instance]) -> Tuple[Dict[str, int], int]:
	deplabel2idx = {}
	deplabels = []
	root = ''
	deplabel2idx[PAD] = len(deplabel2idx)
	deplabels.append(PAD)

	# if self_label not in deplabel2idx:
	# 	deplabels.append(self_label)
	# 	deplabel2idx[self_label] = len(deplabel2idx)
	for inst in insts:
		idx = 0
		for synhead, syndep_label in zip(inst.synheads,inst.syndep_labels):
			if synhead != 0:
				if syndep_label not in deplabels:
					deplabels.append(syndep_label)
					deplabel2idx[syndep_label] = len(deplabel2idx)
			elif root == '':
				root = syndep_label
				deplabels.append(syndep_label)
				deplabel2idx[syndep_label] = len(deplabel2idx)
			elif root != syndep_label:
				print('root = ' + root + ', rel for root = ' + syndep_label)
				inst.syndep_labels[idx] = root
			idx += 1

		# for label in inst.syndep_labels:
		# 	if label not in deplabels:
		# 		deplabels.append(label)
		# 		deplabel2idx[label] = len(deplabel2idx)
	root_dep_label_id = deplabel2idx[root_dep_label]

	print("#dep labels: {}".format(len(deplabels)))
	print("dep label 2idx: {}".format(deplabel2idx))
	return deplabel2idx, root_dep_label_id