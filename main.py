"""Entry point."""
import os

import torch

import data
import config
import utils
import trainer
import trainer2

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.network_type == 'rnn':
        assert args.network_type in ('rnn', 'cnn')
        dataset = data.text.Corpus(args.data_path)
    elif args.dataset == 'cifar':
        args.network_type in ('rnn', 'cnn')
        dataset = data.image.Image(args)
    else:
        args.network_type not in ('rnn', 'cnn')
        dataset = data.tabular.Tabular(args)

    if args.network_type in ('rnn', 'cnn'):
        trnr = trainer.Trainer(args, dataset)
    else:
        trnr = trainer2.Trainer(args, dataset)

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in "
                                      "`derive` mode")
        trnr.derive()
    else:
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a "
                            "pretrained model")
        trnr.test()


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
