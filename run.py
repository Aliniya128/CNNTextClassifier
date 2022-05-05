import argparse
from Trainer import Trainer
from dataset.utils import word2vector

parse = argparse.ArgumentParser('text classifier')
parse.add_argument('--epochs', type=int, help="训练轮次", default=60)
parse.add_argument('--w2v', action='store_true', help="词转化为词向量")
parse.add_argument('--train', action='store_true', help="是否进行训练")
args = parse.parse_args()


if __name__ == "__main__":
    if args.w2v:
        word2vector()
    if args.train:
        trainer = Trainer(args)
        trainer.train()