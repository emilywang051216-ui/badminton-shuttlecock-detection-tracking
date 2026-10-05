# Badminton Shuttlecock Detection and Tracking

A computer vision project for detecting and tracking shuttlecocks in badminton match videos, comparing YOLOv11 for frame-based object detection with TrackNetV2 for motion-aware temporal tracking.

It compares two models. YOLOv11 looks at one frame at a time. TrackNetV2 looks at three frames in a row and predicts a heatmap, so it can use motion. The real question was which one actually holds up on a video it has never seen.

## Data

The dataset is [Shuttlecock](https://universe.roboflow.com/mathieu-cartron/shuttlecock-cqzy3/dataset/1) by Mathieu Cartron on Roboflow Universe. It is 8,053 labelled frames from 3 broadcast match videos, with a single class for the shuttlecock.

## The finding worth talking about

The data is only 3 match videos, and the original split mixed all three into train, val and test. Frames next to each other in a rally look almost the same, so that split leaks. The test score looks amazing and means almost nothing for a new clip.

So we split by video instead, holding one whole video out:

| | YOLOv11 (mAP50) | TrackNetV2 (F1) |
|:--|:--|:--|
| mixed split (leaky) | 0.66 | high but misleading |
| held out video | 0.00 to 0.43 | 0.73 to 0.88 |

YOLO falls apart on an unseen video (0.00 on the hardest fold). TrackNetV2 stays around 0.8, because motion generalizes across broadcasts much better than how a single frame looks. That gap is the whole story.

## Run it

Works on Windows or Linux with a CUDA GPU. Install first:

```
pip install -r requirements.txt
```

Run the whole pipeline at once (needs bash, so Git Bash on Windows or any Linux shell):

```
./run_all.sh
```

Or run the steps yourself, which is the same on any platform:

```
python scripts/01_prepare_data.py
python scripts/02_train_yolo.py --split lovo_test3
python scripts/03_train_tracknet.py --test-video 3
python scripts/04_eval_compare.py
```

Track a new clip with the deployment model:

```
python scripts/05_infer_video.py --model tracknet --source your_clip.mp4 \
    --weights models/3videos_deploy/tracknet_all/best.pt --fps 25
```

The scripts adapt to the machine on their own. Dataloader workers and plotting switch between Windows and Linux automatically, and if your GPU has less memory you can drop the batch size with `--batch` or in `configs/yolo_smallobj.yaml`.

## Two sets of weights

models/ holds two kinds:

1. `2videos_lovo` trained on 2 videos with one held out. These give the honest generalization numbers.
2. `3videos_deploy` trained on all 3 videos. Use these when you actually want to track a new video.

## Layout

```
scripts/    prep, train YOLO, train TrackNet, eval, infer, export, plus shared libs
configs/    training and augmentation settings
run_all.sh  one command that runs the whole thing
```

data/, models/, outputs/ and the dataset zip are large or generated, so they stay out of git.

## Notes

The shuttlecock is tiny (around 5 by 9 pixels) and very fast, which is what makes this a fun small object problem. Inference runs comfortably faster than real time once the model is exported to ONNX.

## Author

Yiqing Wang
The University of Melbourne
