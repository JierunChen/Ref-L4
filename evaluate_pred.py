import argparse
import json
from ref_l4 import RefL4Dataset, RefL4Evaluator

def main(args):
    custom_transforms = None
    ref_l4_dataset = RefL4Dataset(args.dataset_path, split=args.split, custom_transforms=custom_transforms)
    print("Dataset loaded. Length:", len(ref_l4_dataset))

    evaluator = RefL4Evaluator(dataset=ref_l4_dataset)

    with open(args.pred_json_path, 'r') as f:
        predictions = json.load(f)
    evaluator.evaluate(predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the predictions on Ref-L4 dataset.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Ref-L4 dataset.')
    parser.add_argument('--split', type=str, default='all', choices=['val', 'test', 'all'], help='Dataset split to use (val, test, all).')
    parser.add_argument('--pred_json_path', type=str, required=True, help='Path to the predictions JSON file.')

    args = parser.parse_args()

    main(args)
