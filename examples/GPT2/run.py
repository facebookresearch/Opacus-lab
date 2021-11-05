from opacus_lab.models.GPT2.dataset import CorpusDataset
from opacus_lab.models.GPT2.train import set_up_optim, train
from opacus_lab.models.GPT2.refactor import refactor_transformer, test_refactor
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
# until opacus-lab is pip installable as a module we
# work around by just appending a sys path
#import sys
#sys.path.append('../../../opacus-lab')


parser = argparse.ArgumentParser(description="GPT-2 implementation for Opacus")
parser.add_argument(
    "-sr",
    "--sample-rate",
    type=float,
    default=0.001,
    metavar="SR",
    help="sample rate used for batch construction (default: 0.001)",
)
parser.add_argument(
    "-bs",
    "--batch-size",
    type=int,
    default=1,
    metavar="BS",
    help="Batch size (default: 1)",
)
parser.add_argument(
    "-vbs",
    "--virtual-batch-size",
    type=int,
    default=4,
    metavar="VBS",
    help="Virtual batch size (default: 4)",
)
parser.add_argument(
    "--seqlen",
    type=int,
    default=32,
    help="Sequence length to block text into (default: 32)",
)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=4096,
    help="# of warmup steps to take (default: 4096)",
)
parser.add_argument(
    "--size",
    type=str,
    default='S',
    choices=['S', 'L', 'M', 'D'],
    help="Model size of GPT-2 (default: S)",
)
parser.add_argument(
    "--finetune-layers",
    type=int,
    default=-1,
    help="Fine-tune from which layer # upwards? Embeddings are layer # 0 \
    (default: -1)",
)
parser.add_argument(
    "--low-rank-head",
    action="store_true",
    default=False,
    help="Should we use a low-rank output layer? (default: false)",
)
parser.add_argument(
    "--head-rank",
    type=int,
    default=768,
    help="Rank of the output layer. Ignored if --low-rank-head is not set.\
    (default: 768)",
)
parser.add_argument(
    "--perturb",
    action="store_true",
    default=False,
    help="Should we use a low-rank perturbation for the output layer? \
    (default: false)",
)
parser.add_argument(
    "-n",
    "--epochs",
    type=int,
    default=1,
    metavar="N",
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="learning rate (default: .1)",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    metavar="S",
    help="Noise multiplier (default 1.0)",
)
parser.add_argument(
    "-c",
    "--gradclip",
    type=float,
    default=1.0,
    metavar="C",
    help="Clip per-sample gradients to this norm (default 1.0)",
)
parser.add_argument(
    "--delta",
    type=float,
    default=1e-5,
    metavar="D",
    help="Target delta (default: 1e-5)",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="Dropout value (default: 0.0)",
)
parser.add_argument(
    "--max-train-iters",
    type=float,
    default=float('inf'),
    help="Set a max # of training iterations (default: inf)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="GPU ID for this process (default: 'cuda')",
)
parser.add_argument(
    "--save-model",
    action="store_true",
    default=False,
    help="Save the trained model (default: false)",
)
parser.add_argument(
    "--checkpoint-model",
    action="store_true",
    default=True,
    help="Checkpoint the model after each evaluation (default: true)",
)
parser.add_argument(
    "--checkpoint-path",
    type=str,
    default='./',
    help="Path to save model checkpoints",
)
parser.add_argument(
    "--print-freq",
    type=int,
    default=100,
    help="Print update every --print-freq iters  (default: 100)",
)
parser.add_argument(
    "--val-freq",
    type=int,
    default=2000,
    help="Run validation set every --val-freq iters  (default: 2000)",
)
parser.add_argument(
    "--val-iters",
    type=int,
    default=512,
    help="# of validation samples to run  (default: 512)",
)
parser.add_argument(
    "--disable-dp",
    action="store_true",
    default=False,
    help="Disable privacy training",
)
parser.add_argument(
    "--skip-refactor",
    action="store_true",
    default=False,
    help="Skip refactor and use Huggingface's GPT-2. \
    Requires setting --disable-dp flag.",
)
parser.add_argument(
    "--secure-rng",
    action="store_true",
    default=False,
    help="Enable Secure RNG to have trustworthy privacy guarantees. \
    Comes at a performance cost",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1234,
    help="Set random seed. Automatically ignored if using secure RNG.",
)
parser.add_argument(
    "--data-root",
    type=str,
    default="./",
    help="Where wikitext is/will be stored",
)


def _load_model(args):
    if args.size == 'L':
        s = 'gpt2-large'
    elif args.size == 'M':
        s = 'gpt2-medium'
    elif args.size == 'S':
        s = 'gpt2'
    elif args.size == 'XL':
        s = 'gpt2-xl'
    elif args.size == 'D':
        s = 'distilgpt2'
    else:
        raise ValueError(f"Unexpected value of arg.size {args.size}")
    model = GPT2LMHeadModel.from_pretrained(s)
    if not args.skip_refactor:
        pretrained_model = model
        model = refactor_transformer(pretrained_model,
                                     use_low_rank=args.low_rank_head,
                                     size=args.size,
                                     lm_head_rank=args.head_rank,
                                     perturb=args.perturb,
                                     dropout=args.dropout)
        assert test_refactor(pretrained_model, model), 'Refactor failed...'
        print('Refactor successful!')
    return model


def _load_wikitext(path):
    corpus = dict()
    for dset in ['valid', 'train', 'test']:
        corpus[dset] = torch.load(f'{path}/wikitext-103-{dset}-corpus.pt')
    return corpus


def _dataloading(args):
    corpus = _load_wikitext(args.data_root)
    trainloader = DataLoader(
        CorpusDataset(corpus['train'], args.seqlen),
        shuffle=True, batch_size=args.batch_size)
    testloader = DataLoader(
        CorpusDataset(corpus['test'], args.seqlen),
        shuffle=True, batch_size=args.batch_size)
    valloader = DataLoader(
        CorpusDataset(corpus['valid'], args.seqlen),
        shuffle=True, batch_size=args.batch_size)

    return trainloader, testloader, valloader


def _training(args, model, loaders):
    trainloader, testloader, valloader = loaders
    n_samples = len(trainloader.dataset)
    sample_rate = args.virtual_batch_size/n_samples
    optim, scheduler = set_up_optim(
        model, args.device, dp=(not args.disable_dp),
        finetune=args.finetune_layers, batch_size=args.batch_size,
        noise_multiplier=args.sigma, max_grad_norm=args.gradclip,
        alphas=[2, 4, 8, 16, 32], lr=args.lr, sample_rate=sample_rate
        warmup_steps=args.warmup_steps, Huggingface=args.skip_refactor)
    L = dict()
    for e in range(args.epochs):
        L[e] = train(model, args.device, trainloader, valloader, e,
                     optim, args.virtual_batch_size, args.max_train_iters,
                     scheduler, print_freq=args.print_freq,
                     Huggingface=args.skip_refactor,
                     delta=args.delta if not args.disable_dp else 1.0,
                     checkpoint_path=args.checkpoint_path if
                     args.checkpoint_model else None)
    return L


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = _load_model(args)
    loaders = _dataloading(args)
    _training(args, model, loaders)
