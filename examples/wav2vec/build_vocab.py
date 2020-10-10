# coding: utf-8
from collections import defaultdict, Counter
import argparse


def add_args(parser):
    parser.add_argument(
        '--base-vocab-fp',
        type=str,
        default='data/base_vocab.txt',
        help='file path of base vocab from masr',
    )
    parser.add_argument(
        '--corpus-text-fp', type=str,
        default='data/train-text.txt',
        help='文本语料所在文件（每行为一个句子）',
    )
    parser.add_argument(
        '-o', '--output-vocab-fp', type=str,
        default='data/dict.ltr.txt',
        help='更新后的vocab存放在此文件（第一列为字，第二列为对应的字频）',
    )


def read_base_vocab(fp):
    vocab = defaultdict(int)
    with open(fp) as f:
        for line in f:
            c, cnt = line.rstrip('\n').rsplit(maxsplit=1)
            vocab[c] = int(cnt)

    return vocab


def update_vocab_by_corpus(vocab, corpus_fp):
    print(f'before update: corpus size: {len(vocab)}')
    with open(corpus_fp) as f:
        for idx, line in enumerate(f):
            for k, v in Counter(line.rstrip('\n')).items():
                vocab[k] += v

        print(f'{idx} lines from corpus {corpus_fp} are read.')
    print(f'after update: corpus size: {len(vocab)}')
    return vocab


def write_vocab(vocab, out_fp):
    with open(out_fp, 'w') as f:
        vocab['|'] = max(vocab.values()) + 1  # 默认的分隔符，放在最前面
        for c, cnt in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            f.write(f'{c} {cnt}\n')


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    vocab = read_base_vocab(args.base_vocab_fp)
    vocab = update_vocab_by_corpus(vocab, args.corpus_text_fp)
    write_vocab(vocab, args.output_vocab_fp)


if __name__ == '__main__':
    main()
