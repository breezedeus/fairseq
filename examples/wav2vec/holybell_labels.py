# coding: utf-8

"""
针对holybell数据，生成ASR精调时所需的文件。
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import os


def read_tsv_file(fp):
    info_list = []
    with open(fp, "r") as tsv:
        root = next(tsv).strip()
        for line in tsv:
            info_list.append(line.rstrip('\n'))

    return root, info_list


def read_label_file(fp):
    labels = dict()
    with open(fp) as f:
        for line in f:
            user_id, asr_txt, wav_fp = line.rstrip('\n').split('\t')
            labels[wav_fp] = asr_txt
    print(f'{len(labels)} labels are read')
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--label-fp", required=True, help='asr结果所在的文件')
    parser.add_argument(
        "--output-tsv", required=True, help='清理后的tsv。原始tsv中的某些文件可能不存在asr结果，去掉这些异常文件'
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = read_label_file(args.label_fp)

    root, info_list = read_tsv_file(args.tsv)
    with open(args.output_tsv, "w") as tsv_out, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        print(root, file=tsv_out)
        for line in info_list:
            wav_fp = line.split()[0]
            if wav_fp not in labels:
                continue
            print(line, file=tsv_out)
            asr_txt = labels[wav_fp]
            print(asr_txt, file=wrd_out)
            print(
                " ".join(list(asr_txt.replace(" ", "|"))) + " |", file=ltr_out,
            )


if __name__ == "__main__":
    main()
