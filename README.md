# Ref-L4: a New REC Benchmark in the LMM Era

This repository contains the data loader and evaluation code for our [Ref-L4](https://huggingface.co/datasets/JierunChen/Ref-L4), a new REC benchmark in the LMM era. Check out our paper [Revisiting Referring Expression Comprehension Evaluation in the Era of Large Multimodal Models](https://arxiv.org/abs/2406.16866) for more details

## Introduction

Referring expression comprehension (REC) involves localizing a target based on a textual description. Recent advancements with large multimodal models (LMMs) like CogVLM have achieved high accuracy (92.44% on RefCOCO). However, existing benchmarks (RefCOCO, RefCOCO+, RefCOCOg) have high labeling error rates (14%, 24%, and 5% respectively), undermining evaluations. We address this by excluding problematic instances and reevaluating LMMs, showing significant accuracy improvements. We also introduce Ref-L4, a new REC benchmark with:

- A substantial sample size with 45,341 annotations
- A diverse range of object categories with 365 distinct types and varying instance scales from 30 to 3,767
- Lengthy referring expressions averaging 24.2 words
- An extensive vocabulary comprising 22,813 unique words

### Ref-L4 examples

<img src="figure/examples.png"  align = "center"  width="800" />

### Labeling errors in RefCOCO, +, g

In the REC task, a referring expression should uniquely describe an instance, which is represented by an accurate bounding box. We have identified and visualized three common types of labeling errors in the RefCOCO, RefCOCO+, and RefCOCOg benchmarks: (a) non-unique referring expressions, which refer to multiple instances within the same image; (b) inaccurate bounding boxes; and (c) misalignment between target instances and their referring expressions, where the referring expressions are either ambiguous or do not refer to any instance in the image.

<img src="figure/error_samples.png"  align = "center"  width="800" />

## Dataset Download

The [Ref-L4 dataset](https://huggingface.co/datasets/JierunChen/Ref-L4) can be downloaded from Hugging Face.

## Installation

You can install the data loader and evaluation API with the following command:

```bash
pip install ./
```

## Data Loader

We provide the `RefL4Dataset` class, which inherits from `torch.utils.data.Dataset`. It accepts the following arguments during initialization:

```txt
- dataset_path (str): Path to the dataset directory.
- split (str): Dataset split, typically "val", "test", or "all".
- images_file (str): Name of the tar file containing images, default to "images.tar.gz".
- custom_transforms: Custom image transformations to apply, default to "None".
```

Additionally, we offer the `change_split` method within the class, which accepts a `split` parameter and is used to switch the dataset split.

## Evaluation

### Evaluation API Introduction

We provide the `RefL4Evaluator` class, which takes the following arguments:

```txt
- dataset (RefL4Dataset): The RefL4Dataset dataset for evaluation.
- split (str): The split of the dataset to evaluate. If None, use the dataset's split. Default is None.
- ann_level_acc_ths (List[float]): The thresholds to evaluate the annotation level accuracy. Default is [0.5, 0.75, 0.9].
- ann_level_macc_ths (List[float]): The thresholds to evaluate the annotation level mAcc. Default is [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
- size_level_acc_ths (List[float]): The thresholds to evaluate the size level accuracy. Default is [0.5, ].
- size_level_macc_ths (List[float]): The thresholds to evaluate the size level mAcc. Default is [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
- small_size_th (int): The threshold to define the small size bbox. Default is 128.
- large_size_th (int): The threshold to define the large size bbox. Default is 256.
- avg_cls_level_acc_ths (List[float]): The thresholds to evaluate the average of all class level accuracy. Default is [0.5, ].
- avg_cls_level_macc_ths (List[float]): The thresholds to evaluate the average of all class level mAcc. Default is [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
```

To evaluate predictions, the `evaluate` method is called, which accepts two arguments: `predictions` and `save_file`. The `predictions` should be a list of dictionaries, each containing three keys: `[id, pred_bbox, format]`. The `id` is the annotation ID, and the `format` specifies the format of `pred_bbox`, which should be either `xyxy` or `xywh`.
A sample prediction file the can be found [here](./demo_models/cogvlm_grounding_pred.json), which is obtained by running the CogVLM-Grounding model.

### Evaluation Output

The evaluation considers three parts:

1. The accuracy under various IoU thresholds and the average accuracy of IoU from 0.5 to 0.95 with a stride of 0.05.
2. The accuracy of small, medium, and large objects.
3. The average accuracy across different classes.

Here is an example output for the predictions from CogVLM-Grounding on the "all" splits:

```txt
Item for split all                    | Value
--------------------------------------------------------------------------------
Ann-level acc iou 0.5                 | 81.69868331091065
Ann-level acc iou 0.75                | 70.76597340155709
Ann-level acc iou 0.9                 | 48.351381751615534
Ann-level macc iou 0.5:0.95           | 66.08808804393375
Ann-level accs for copy               | 81.7, 70.77, 48.35, 66.09
--------------------------------------------------------------------------------
Small acc iou 0.5                     | 75.0561797752809
Small macc iou 0.5:0.95               | 52.853932584269664
Medium acc iou 0.5                    | 86.43470790378007
Medium macc iou 0.5:0.95              | 71.31099656357388
Large acc iou 0.5                     | 77.90972003774772
Large macc iou 0.5:0.95               | 66.25196602705253
Size level accs for copy              | 75.06, 52.85, 86.43, 71.31, 77.91, 66.25
--------------------------------------------------------------------------------
Average class-level acc iou 0.5       | 72.42029130234654
Average class-level macc iou 0.5:0.95 | 52.56151912966669
Avg class-level accs for copy         | 72.42, 52.56
```


To reproduce this result, you can run:

```bash
python eval_pred.py \
    --dataset_path <path to Ref-L4 folder> \
    --split all \
    --pred_json_path ./demo_models/cogvlm_grounding_pred.json
```

## Dataset License

The Ref-L4 dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license](https://creativecommons.org/licenses/by-nc/4.0/). Please note that the images in the Ref-L4 dataset are derived from the following datasets, each with their respective licenses:
- **RefCOCO**: Licensed under the [Apache-2.0 license](http://www.apache.org/licenses/).
- **RefCOCO+**: Licensed under the [Apache-2.0 license](http://www.apache.org/licenses/).
- **RefCOCOg**: Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).
- **COCO 2014**: Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).
- **Objects365**: Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](http://creativecommons.org/licenses/by/4.0/).

By using the Ref-L4 dataset, you agree to comply with the licensing terms of these source datasets.
