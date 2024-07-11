# Copyright (c) 2024 Jierun Chen. All rights reserved.
import torch
from statistics import mean
import pandas as pd
import math
from datasets import load_dataset
from .dataloader import RefL4Dataset
from .overlaps import bbox_overlaps
import numpy as np

mapping_dict = {
    "refcoco_1": "o365_1",
    "refcoco_10": "o365_41",
    "refcoco_11": "o365_177",
    "refcoco_13": "o365_128",
    "refcoco_14": "o365_250",
    "refcoco_15": "o365_25",
    "refcoco_16": "o365_56",
    "refcoco_17": "o365_140",
    "refcoco_18": "o365_93",
    "refcoco_19": "o365_79",
    "refcoco_2": "o365_47",
    "refcoco_20": "o365_100",
    "refcoco_21": "o365_97",
    "refcoco_22": "o365_145",
    "refcoco_23": "o365_296",
    "refcoco_24": "o365_179",
    "refcoco_25": "o365_181",
    "refcoco_27": "o365_39",
    "refcoco_28": "o365_40",
    "refcoco_3": "o365_6",
    "refcoco_31": "o365_13",
    "refcoco_32": "o365_44",
    "refcoco_33": "o365_194",
    "refcoco_34": "o365_220",
    "refcoco_35": "o365_119",
    "refcoco_36": "o365_174",
    "refcoco_37": "o365_100000",    # This is a placeholder, the actual value is not provided
    "refcoco_38": "o365_155",
    "refcoco_39": "o365_138",
    "refcoco_4": "o365_59",
    "refcoco_40": "o365_114",
    "refcoco_41": "o365_146",
    "refcoco_42": "o365_147",
    "refcoco_43": "o365_205",
    "refcoco_44": "o365_9",
    "refcoco_46": "o365_36",
    "refcoco_47": "o365_11",
    "refcoco_48": "o365_89",
    "refcoco_49": "o365_85",
    "refcoco_5": "o365_115",
    "refcoco_50": "o365_94",
    "refcoco_51": "o365_26",
    "refcoco_52": "o365_113",
    "refcoco_53": "o365_83",
    "refcoco_54": "o365_266",
    "refcoco_55": "o365_104",
    "refcoco_56": "o365_142",
    "refcoco_57": "o365_153",
    "refcoco_58": "o365_235",
    "refcoco_59": "o365_144",
    "refcoco_6": "o365_56",
    "refcoco_60": "o365_151",
    "refcoco_61": "o365_98",
    "refcoco_62": "o365_3",
    "refcoco_63": "o365_51",
    "refcoco_64": "o365_26",
    "refcoco_65": "o365_76",
    "refcoco_67": "o365_98",
    "refcoco_7": "o365_117",
    "refcoco_70": "o365_154",
    "refcoco_72": "o365_37",
    "refcoco_73": "o365_74",
    "refcoco_74": "o365_116",
    "refcoco_75": "o365_133",
    "refcoco_76": "o365_107",
    "refcoco_77": "o365_62",
    "refcoco_78": "o365_164",
    "refcoco_79": "o365_135",
    "refcoco_8": "o365_66",
    "refcoco_80": "o365_278",
    "refcoco_81": "o365_82",
    "refcoco_82": "o365_134",
    "refcoco_84": "o365_19",
    "refcoco_85": "o365_95",
    "refcoco_86": "o365_31",
    "refcoco_87": "o365_170",
    "refcoco_88": "o365_70",
    "refcoco_89": "o365_328",
    "refcoco_9": "o365_22",
    "refcoco_90": "o365_227",
}


class RefL4Evaluator:
    def __init__(
            self,
            dataset: RefL4Dataset,
            split=None, 
            ann_level_acc_ths=[0.5, 0.75, 0.9],
            ann_level_macc_ths=[i/100 for i in range(50,100,5)],
            size_level_acc_ths=[0.5, ],
            size_level_macc_ths=[i/100 for i in range(50,100,5)],
            small_size_th=128,
            large_size_th=256,
            avg_cls_level_acc_ths=[0.5, ],
            avg_cls_level_macc_ths=[i/100 for i in range(50,100,5)],
        ) -> None:
        '''
        dataset (RefL4Dataset): The RefL4Dataset dataset for evaluation.
        split (str): The split of the dataset to evaluate. If None, use the dataset's split. Default is None.
        ann_level_acc_ths (List[float]): The thresholds to evaluate the annotation level accuracy. Default is [0.5, 0.75, 0.9].
        ann_level_macc_ths (List[float]): The thresholds to evaluate the annotation level mAcc. Default is [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
        size_level_acc_ths (List[float]): The thresholds to evaluate the size level accuracy. Default is [0.5, ].
        size_level_macc_ths (List[float]): The thresholds to evaluate the size level mAcc. Default is [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
        small_size_th (int): The threshold to define the small size bbox. Default is 128.
        large_size_th (int): The threshold to define the large size bbox. Default is 256.
        avg_cls_level_acc_ths (List[float]): The thresholds to evaluate the average of all class level accuracy. Default is [0.5, ].
        avg_cls_level_macc_ths (List[float]): The thresholds to evaluate the average of all class level mAcc. Default is [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
        '''
        self.dataset = dataset.dataset
        self.split=split
        if split is None:
            self.split=dataset.split

        self.ann_level_acc_ths = ann_level_acc_ths
        self.ann_level_macc_ths = ann_level_macc_ths
        self.size_level_acc_ths = size_level_acc_ths
        self.size_level_macc_ths = size_level_macc_ths
        self.avg_cls_level_acc_ths = avg_cls_level_acc_ths
        self.avg_cls_level_macc_ths = avg_cls_level_macc_ths
        
        self.small_size_th=small_size_th
        self.large_size_th=large_size_th

    @staticmethod
    def calculate_iou_acc(bboxes_1, bboxes_2, thresh=0.5):
        """
        bboxes_1 (torch.Tensor, numpy.Array): shape=[N,4], format=[x1, y1, x2, y2]
        bboxes_2 (torch.Tensor, numpy.Array): shape=[N,4], format=[x1, y1, x2, y2]
        calculate the iou and acc of the pred_bboxes and gt_bboxes,
        if iou(pred_bboxes[i],gt_bboxes[i])>0.5, then acc+=1
        all pred_bboxes_i and gt_bboxes_i are one to one assigned.
        
        """
        iou=bbox_overlaps(bboxes_1,bboxes_2,mode='iou', is_aligned=True)
        if(type(thresh) is not list):
            thresh=[thresh]
        accs=dict()
        for t in thresh:
            accs[t]=(iou>t).sum().item()/len(iou)
        return iou,accs

    def evaluate(self, predictions, save_file=None):
        """
        Evaluate given dataset and predictions.

        Parameters:
        - predictions (List(Dict)): The predictions to evaluate. 
            Each item in the list is a dict, containing the keys: 'pred_bbox', 'id' and 'format', 
            where 'id' is the annotation id, 'format' is the bbox format 'xyxy' or 'xywh'.
            e.g.:
            [
                {
                'pred_bbox': [x1, y1, x2, y2],
                'id': '000000',
                'format': 'xyxy'
                },
                ...
            ]
        - save_file (str): The file to save the evaluation results to.
        """
        if(len(predictions)==0):
            print("Warning: No predictions found.")
            return dict()
        
        # convert predictions to a dict, key is the id, raise error if there are duplicate ids
        predictions_dict={pred['id']:pred for pred in predictions}
        if(len(predictions)!=len(predictions_dict)):
            raise ValueError("Duplicate ids found in the predictions.")

        gt_bboxes = []
        pred_bboxes = []
        dataset_split = self.dataset[self.split]

        for idx, gt_data in enumerate(dataset_split):
            gt_bbox = gt_data['bbox']
            gt_bboxes.append([gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]])
            
            # raise error if the id canot be found in the predictions
            if gt_data['id'] not in predictions_dict:
                raise ValueError(f"Id {gt_data['id']} not found in the predictions.")

            pred_bbox = predictions_dict[gt_data['id']]['pred_bbox']
            if predictions_dict[gt_data['id']]['format'] == 'xywh':
                pred_bbox = [pred_bbox[0], pred_bbox[1], pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]]
            pred_bboxes.append(pred_bbox)


        # calculate_iou
        iou = bbox_overlaps(torch.tensor(gt_bboxes), torch.tensor(pred_bboxes), mode='iou', is_aligned=True).numpy()

        acc_all = dict()

        # Annotation level evaluation
        for th in self.ann_level_acc_ths:
            acc = (iou > th).sum().item() / len(iou)
            key = f"Ann-level acc iou {th}"
            acc_all[key] = acc * 100
        
        macc = []
        for th in self.ann_level_macc_ths:
            acc = (iou > th).sum().item() / len(iou)
            macc.append(acc)
        key = f"Ann-level macc iou {self.ann_level_macc_ths[0]}:{self.ann_level_macc_ths[-1]}"
        acc_all[key] = mean(macc) * 100

        # get the acc for copy, so that we can output the results as a table, round to 2 decimal places
        # acc_all['Ann-level accs for copy'] = [acc_all[key] for key in acc_all]
        acc_all['Ann-level accs for copy'] = [
            round(acc_all[key], 2) for key in acc_all
        ]

        # Size evaluation
        small_size_list = []
        medium_size_list = []
        large_size_list = []
        for idx, gt_data in enumerate(dataset_split):
            gt_bbox = gt_data['bbox']
            obj_size = math.sqrt(gt_bbox[2]*gt_bbox[3])
            iou_item = iou[idx]

            # gather the small, medium and large size bboxes
            if obj_size < self.small_size_th:
                small_size_list.append(iou_item)
            elif obj_size <= self.large_size_th:
                medium_size_list.append(iou_item)
            else:
                large_size_list.append(iou_item)
        
        #  small size evaluation
        for th in self.size_level_acc_ths:
            acc_small = sum(i > th for i in small_size_list) / len(small_size_list) * 100
            key = f"Small acc iou {th}"
            acc_all[key] = acc_small 
                
        macc_small = []
        for th in self.size_level_macc_ths:
            small_size_list = np.array(small_size_list)
            acc_small = (small_size_list > th).sum().item() / len(small_size_list) * 100
            macc_small.append(acc_small)
        key = f"Small macc iou {self.size_level_macc_ths[0]}:{self.size_level_macc_ths[-1]}"
        acc_all[key] = mean(macc_small)

        # medium size evaluation
        for th in self.size_level_acc_ths:
            acc_medium = sum(i > th for i in medium_size_list) / len(medium_size_list) * 100
            key = f"Medium acc iou {th}"
            acc_all[key] = acc_medium

        macc_medium = []
        for th in self.size_level_macc_ths:
            medium_size_list = np.array(medium_size_list)
            acc_medium = (medium_size_list > th).sum().item() / len(medium_size_list) * 100
            macc_medium.append(acc_medium)
        key = f"Medium macc iou {self.size_level_macc_ths[0]}:{self.size_level_macc_ths[-1]}"
        acc_all[key] = mean(macc_medium)

        # large size evaluation
        for th in self.size_level_acc_ths:
            acc_large = sum(i > th for i in large_size_list) / len(large_size_list) * 100
            key = f"Large acc iou {th}"
            acc_all[key] = acc_large

        macc_large = []
        for th in self.size_level_macc_ths:
            large_size_list = np.array(large_size_list)
            acc_large = (large_size_list > th).sum().item() / len(large_size_list) * 100
            macc_large.append(acc_large)
        key = f"Large macc iou {self.size_level_macc_ths[0]}:{self.size_level_macc_ths[-1]}"
        acc_all[key] = mean(macc_large)

        # get the size-level acc for copy, so that we can output the results as a table, round to 2 decimal places
        acc_all['Size level accs for copy'] = [
            round(acc_all[key], 2) for key in acc_all
            if 'Small' in key or 'Medium' in key or 'Large' in key
        ]

        # Average class-level evaluation
        iou_avg_cls_level_acc_ths = dict()
        for idx, gt_data in enumerate(dataset_split):
            iou_item = iou[idx]
            if gt_data['ori_category_id'] in mapping_dict:
                ori_category_id = mapping_dict[gt_data['ori_category_id']]
            else:
                ori_category_id = gt_data['ori_category_id']

            if ori_category_id not in iou_avg_cls_level_acc_ths:
                iou_avg_cls_level_acc_ths[ori_category_id] = []
            iou_avg_cls_level_acc_ths[ori_category_id].append(iou_item)
        
        for th in self.avg_cls_level_acc_ths:
            acc_list = []
            for key in iou_avg_cls_level_acc_ths:
                iou_array = np.array(iou_avg_cls_level_acc_ths[key])
                acc = (iou_array > th).sum().item() / len(iou_array) * 100
                acc_list.append(acc)
            key = f"Average class-level acc iou {th}"
            acc_all[key] = mean(acc_list)

        # macc
        macc_list = []
        for th in self.avg_cls_level_macc_ths:
            acc_list = []
            for key in iou_avg_cls_level_acc_ths:
                iou_array = np.array(iou_avg_cls_level_acc_ths[key])
                acc = (iou_array > th).sum().item() / len(iou_avg_cls_level_acc_ths[key]) * 100
                acc_list.append(acc)
            macc_list.append(mean(acc_list))
        key = f"Average class-level macc iou {self.avg_cls_level_macc_ths[0]}:{self.avg_cls_level_macc_ths[-1]}"
        acc_all[key] = mean(macc_list)

        # get the avg_cls-level acc for copy, so that we can output the results as a table, round to 2 decimal places
        acc_all['Avg class-level accs for copy'] = [
            round(acc_all[key], 2) for key in acc_all
            if 'Average class-level' in key
        ]


        # Output as table
        table = []
        table.append([f"Item for split {self.split}", "Value"])
        for k, v in acc_all.items():
            if isinstance(v, list):
                table.append([k, ", ".join(map(str, v))])
            else:
                table.append([k, v])

        # Define where to add horizontal lines
        horizontal_lines = {1, 6, 13}  # After header, IoU, and Subject evaluations

        # Print table with selective horizontal lines
        max_len = max(len(row[0]) for row in table)
        for i, row in enumerate(table):
            if i in horizontal_lines:
                print('-' * (max_len + 3 + max(len(str(r[1])) for r in table)))
            print(f"{row[0].ljust(max_len)} | {row[1]}")

        if(save_file is not None):
            acc_all.pop('Ann-level accs for copy')
            acc_all.pop('Size level accs for copy')
            acc_all.pop('Avg class-level accs for copy')
            df=pd.DataFrame(acc_all, index=[0])
            df.to_csv(save_file)        
                     
        return acc_all

# Example usage:
if __name__ == '__main__':
    from dataloader import RefL4Dataset
    import json
    custom_transforms = None
    ref_l4_dataset = RefL4Dataset('/Users/jchen12/Documents/misc/Ref-L4', split='all', custom_transforms=custom_transforms)
    print("Dataset loaded. Length:", len(ref_l4_dataset))

    evaluator = RefL4Evaluator(dataset=ref_l4_dataset)

    pred_json_path = "/Users/jchen12/Documents/misc/cogvlm_grounding_pred.json"
    with open(pred_json_path, 'r') as f:
        predictions = json.load(f)
    evaluator.evaluate(predictions)