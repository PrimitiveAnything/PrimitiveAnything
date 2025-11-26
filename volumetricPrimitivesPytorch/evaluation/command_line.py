from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='iou_evaluation')

    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gt_dir', type=str, default=None)
    parser.add_argument('--result_file', type=str, default='results.txt')
    args = parser.parse_args()
    return args


