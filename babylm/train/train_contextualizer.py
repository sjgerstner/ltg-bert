#training code for ltg-bert + contextualizer
import argparse
from itertools import count

import torch

#from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk, Sequence, Value
from datasets.distributed import split_dataset_by_node

from train import *

def parse_arguments():
    """overrides the method from train.py"""#TODO
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default='$SCRATCH/babylm_100M/Contextualizer_ltgbert', type=str, help="The input data dir with the contextualized dataset.")
    parser.add_argument("--config_file", default="../configs/base.json", type=str, help="The BERT model config")
    parser.add_argument("--output_dir", default="../checkpoints/base", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--vocab_path", default="../tokenizer.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to a previous checkpointed training state.")
    parser.add_argument("--model_type", choices=['ltgbert', 'elcbert'], default='ltgbert', type=str, help="The type of model to train.")

    # Other parameters
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--scheduler", default="cosine", type=str)
    parser.add_argument("--seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=1024, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=1.0e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=1, type=int, help="Total number of epochs.")#TODO adapt
    parser.add_argument("--long_after", default=1, type=float) 
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--log_freq', type=int, default=10, help='frequency of logging loss.')
    parser.add_argument("--mask_p", default=0.15, type=float, help="Masking probability.")
    parser.add_argument("--short_p", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--activation_checkpointing', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    return args

if __name__=="__main__":
    args = parse_arguments()

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        _, initial_epoch, global_step = checkpoint["args"], checkpoint["epoch"] + 1, checkpoint["global_step"]
        # args = vars(args).copy()
        # args.update(vars(checkpoint_args))
        # args = argparse.Namespace(**args)
    else:
        checkpoint, initial_epoch, global_step = None, 0, 0
        
    print('loading training data...')
    train_data = load_from_disk(args.input_path).with_format("torch")['train']
    len_full_data = len(train_data)
    
    if is_main_process():
        print('computing max steps...')
    print('len(train_data):', len(train_data))
    print('args.batch_size:', args.batch_size)
    print('args.epochs:', args.epochs)
    min_length = (len_full_data // torch.cuda.device_count()) // args.batch_size #assumes all gpus on same node! Otherwise move this block into setup_training().
    print('min_length = len(train_data) // torch.cuda.device_count() // args.batch_size =', min_length)
    args.max_steps = (min_length // args.gradient_accumulation_steps) * args.epochs +1
    print('args.max_steps = min_length // args.gradient_accumulation_steps * args.epochs + 1 = ', args.max_steps)
    
    print('setting up training...')
    device, local_rank = setup_training(args)
    print('done')

    if is_main_process():
        print('loading tokenizer...')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_path)
    tokenizer.mask_token = '[MASK]'
    tokenizer.pad_token = '[PAD]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.sep_token = '[SEP]'
    if is_main_process():
        print(tokenizer)
    
    #train_data = split_dataset_by_node(full_data, get_rank(), get_world_size(),)
    train_data = train_data.cast_column("attention_mask", Sequence(Value("bool")))
    if is_main_process():
        print(train_data)
        print(train_data[1000000]['input_ids'])
        print(tokenizer.decode(train_data[1000000]['input_ids']))

    if is_main_process():
        print('preparing model and optimizer...')
    model, config, optimizer, scheduler, grad_scaler = prepare_model_and_optimizer(args, device, local_rank, checkpoint)
    if is_main_process():
        print('done')
        print('adapting min length...')
    #min_length = torch.tensor(min_length, dtype=torch.long, device=device)
    #torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)
    if is_main_process():
        print('done')

    for epoch in count(initial_epoch):
        if is_main_process():
            print('epoch', epoch, '...')
        global_step = training_epoch(model, train_data, optimizer, scheduler, grad_scaler, global_step, epoch, args, device, min_length, tokenizer=tokenizer)
        if is_main_process():
            print('done')
        if (epoch+1) % 2 == 0:
            checkpoint_path = save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args)
        epoch+=1
        if epoch >= args.epochs:
            break

    # works because epoch exists until the function exits
    checkpoint_path = save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args)
