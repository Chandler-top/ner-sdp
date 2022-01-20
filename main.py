# sys and pytorch package
import argparse
import random
import numpy as np
import torch
import os
import pickle
from torch.utils.data import DataLoader
from collections import Counter
# huggingface and other packages
from tqdm import tqdm
import time
import tarfile
from transformers import AutoTokenizer, AutoModel
from transformers import ElectraModel, ElectraConfig
# self packages
from config import Config
from transformers_util import *
from transformers_dataset import TransformersNERDataset
from transformers_nersdp import TransformersCRF
from utils import *
from eval import *

os.environ["TOKENIZERS_PARALLELISM"] = "False"

def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--dataset', type=str, default="ontonotes")
    parser.add_argument('--mtl mode', type=int, default="0", choices=[0, 1, 2, 3, 4, 5],
                        help="multitask mode, 0 means ner without dp. 1 means ner with syntactic dp. "
                             "2 means ner with semantic dp. 3 means ner with syntactic and semantic dp."
                             "4 means syntactic dp. 5 means semantic dp.")
    parser.add_argument('--optimizer', type=str, default="adamw",
                        help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10,
                        help="default batch size is 30 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=200, help="Usually we set to 100.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=80,
                        help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default=str(time.time()), help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF. if we set to 0 means not using LSTM encoder.")
    parser.add_argument('--num_lstm_layer', type=int, default=0, help="number of lstm layers")
    parser.add_argument('--dropout_emb', type=float, default=0.33, help="dropout for lstm hidden")
    parser.add_argument('--embedder_type', type=str, default="roberta-base",
                        choices=list(context_models.keys()),
                        help="you can use 'bert-base-uncased' and so on")

    parser.add_argument('--dep_emb_size', type=int, default=50, help="embedding size of dependency")
    parser.add_argument('--mlp_arc_size', type=int, default=500, help="hidden size of mlp_arc_size")
    parser.add_argument('--mlp_lab_size', type=int, default=100, help="hidden size of mlp_label_size")
    parser.add_argument('--dropout_mlp_hidden', type=float, default=0.33, help="dropout for lstm hidden")

    # parser.add_argument('--parallel_embedder', type=int, default=0, choices=[0, 1],
    #                     help="use parallel training for those (BERT) models in the transformers. Parallel on GPUs")
    parser.add_argument('--add_iobes_constraint', type=int, default=0, choices=[0,1],
                        help="add IOBES constraint for transition parameters to enforce valid transitions")
    parser.add_argument("--print_detail_f1", type= int, default= 0, choices= [0, 1],
                        help= "Open and close printing f1 scores for each tag after each evaluation epoch")
    parser.add_argument("--earlystop_atr", type=str, default="micro", choices= ["micro", "macro"],
                        help= "Choose between macro f1 score and micro f1 score for early stopping evaluation")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help="training model or test mode")
    parser.add_argument('--test_file', type=str, default="data/conll2003_sample/test.txt",
                        help="test file for test mode, only applicable in test mode")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def train_model(config: Config, epoch: int, train_loader: DataLoader, dev_loader: DataLoader, test_loader: DataLoader):
    ### Data Processing Info
    train_num = len(train_loader)
    print(f"[Data Info] number of training instances: {train_num}")

    print(
        colored(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}", 'red'))
    print(colored(f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface.", 'red'))
    print(colored(f"[Optimizer Info]: Change the optimier in transformers_util.py if you want to make some modifications.", 'red'))
    model = TransformersCRF(config)
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(train_loader) * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)
    print(colored(f"[Optimizer Info] Modify the optimizer info as you need.", 'red'))
    print(optimizer)

    model.to(config.device)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    if os.path.exists("model_files/" + model_folder):
        raise FileExistsError(
            f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
            f"to avoid override.")
    model_path = f"model_files/{model_folder}/lstm_crf.m"
    config_path = f"model_files/{model_folder}/config.conf"
    res_path = f"{res_folder}/{model_folder}.results"
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    os.makedirs(f"model_files/{model_folder}", exist_ok= True) ## create model files. not raise error if exist
    os.makedirs(res_folder, exist_ok=True)
    no_incre_dev = 0
    print(colored(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs",'red'))
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        model.train()
        for iter, batch in tqdm(enumerate(train_loader, 1), desc="--training batch", total=len(train_loader)):
            if iter == 227:
                print ("error")
            optimizer.zero_grad()
            # 统一的loss 输出，同时这个地方要把heads，rels传输去做预测和损失计算
            loss = model(words = batch.input_ids.to(config.device), word_seq_lens = batch.word_seq_len.to(config.device),
                    orig_to_tok_index = batch.orig_to_tok_index.to(config.device), input_mask = batch.attention_mask.to(config.device),
                    synhead_ids = batch.synhead_ids, synlabel_ids = batch.synlabel_ids,
                    labels = batch.label_ids.to(config.device))
            epoch_loss += loss.item()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_loader, "dev", dev_loader.dataset.insts)
        test_metrics = evaluate_model(config, model, test_loader, "test", test_loader.dataset.insts)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            torch.save(model.state_dict(), model_path)
            # Save the corresponding config as well.
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()
            write_results(res_path, test_loader.dataset.insts)
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

    print("Archiving the best Model...")
    with tarfile.open(f"model_files/{model_folder}.tar.gz", "w:gz") as tar:
        tar.add(f"model_files/{model_folder}", arcname=os.path.basename(model_folder))

    print("Finished archiving the models")

    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # evaluate_model(config, model, test_loader, "test", test_loader.dataset.insts)
    write_results(res_path, test_loader.dataset.insts)

def evaluate_model(config: Config, model: TransformersCRF, data_loader: DataLoader, name: str, insts: List, print_each_type_metric: bool = False):
    ## evaluation
    p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
    batch_size = data_loader.batch_size
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(data_loader, 0), desc="--evaluating batch", total=len(data_loader)):
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = model.decode(words= batch.input_ids.to(config.device),
                    word_seq_lens = batch.word_seq_len.to(config.device),
                    orig_to_tok_index = batch.orig_to_tok_index.to(config.device),
                    input_mask = batch.attention_mask.to(config.device))
            batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts, batch_max_ids, batch.label_ids, batch.word_seq_len, config.idx2labels)
            p_dict += batch_p
            total_predict_dict += batch_predict
            total_entity_dict += batch_total
            batch_id += 1
    f1Scores = []
    if print_each_type_metric or config.print_detail_f1 or (config.earlystop_atr == "macro"):
        for key in total_entity_dict:
            precision_key, recall_key, fscore_key = get_metric(p_dict[key], total_entity_dict[key], total_predict_dict[key])
            print(f"[{key}] Prec.: {precision_key:.2f}, Rec.: {recall_key:.2f}, F1: {fscore_key:.2f}")
            f1Scores.append(fscore_key)
        if len(f1Scores) > 0:
            print(f"[{name} set Total] Macro F1: {sum(f1Scores) / len(f1Scores):.2f}")

    total_p = sum(list(p_dict.values()))
    total_predict = sum(list(total_predict_dict.values()))
    total_entity = sum(list(total_entity_dict.values()))
    precision, recall, fscore = get_metric(total_p, total_entity, total_predict)
    print(colored(f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, Micro F1: {fscore:.2f}", 'blue'), flush=True)

    if config.earlystop_atr == "macro" and len(f1Scores) > 0:
        fscore = sum(f1Scores) / len(f1Scores)

    return [precision, recall, fscore]

def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    set_seed(opt, conf.seed)

    if opt.mode == "train":
        print(colored(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer", "blue"))
        tokenizer = context_models[conf.embedder_type]["tokenizer"].from_pretrained(conf.embedder_type, add_prefix_space=True)
        print(colored(f"[Data Info] Reading dataset from: \n{conf.train_file}\n{conf.dev_file}\n{conf.test_file}", "blue"))
        train_dataset = TransformersNERDataset(conf.train_file, tokenizer, number=conf.train_num,
                                               is_train=True)
        conf.label2idx = train_dataset.label2idx
        conf.idx2labels = train_dataset.idx2labels
        conf.lab_size = len(train_dataset.syndep_label2idx)
        conf.rel_size = len(train_dataset.syndep_label2idx)

        dev_dataset = TransformersNERDataset(conf.dev_file, tokenizer, number=conf.dev_num,
                                             label2idx=train_dataset.label2idx, syndep_label2idx=train_dataset.syndep_label2idx,
                                             is_train=False)
        test_dataset = TransformersNERDataset(conf.test_file, tokenizer, number=conf.test_num,
                                              label2idx=train_dataset.label2idx, syndep_label2idx=train_dataset.syndep_label2idx,
                                              is_train=False)
        num_workers = 0
        conf.label_size = len(train_dataset.label2idx)


        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
                                      collate_fn=train_dataset.collate_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=dev_dataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=test_dataset.collate_fn)

        train_model(conf, conf.num_epochs, train_dataloader, dev_dataloader, test_dataloader)
    else:
        folder_name = f"model_files/{opt.model_folder}"
        device = torch.device(opt.device)
        assert os.path.isdir(folder_name)
        f = open(folder_name + "/config.conf", 'rb')
        saved_config = pickle.load(f) # we use `label2idx` from old config, but test file, test number
        f.close()

if __name__ == "__main__":
    main()
