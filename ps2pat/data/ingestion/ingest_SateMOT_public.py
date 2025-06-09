import argparse
import csv
import configparser
import datetime
import glob
import os

from PIL import Image
from pathlib import Path

import gluoncv.torch.data.gluoncv_motion_dataset.dataset as gcvdataset
gcvdataset.__version__ = "1.0"

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import GluonCVMotionDataset, DataSample, AnnoEntity, FieldNames, SplitNames
from gluoncv.torch.data.gluoncv_motion_dataset.utils.ingestion_utils import process_dataset_splits

SATEMOT_LABEL_MAP = {
    1: "airplane",
    2: "ship",
}

def sample_from_mot_csv(csv_path, fps, sample=None, has_gt=True):
    if sample is None:
        id_ = Path(csv_path).stem
        sample = DataSample(id_)
    else:
        sample = sample.get_copy_without_entities()
    with open(csv_path, newline='') as f:
        reader = csv.reader(f, delimiter=',')

        def coord(x):
            return round(float(x))

        for row in reader:
            frame_num = int(row[0])
            obj_id = row[1]
            x = coord(row[2])
            y = coord(row[3])
            w = coord(row[4])
            h = coord(row[5])
            conf = float(row[6])
            if has_gt:
                label = int(row[7])
                visibility = float(row[8])
            else:
                label = int(row[7])
                visibility = 1

            label_text = SATEMOT_LABEL_MAP[label]

            time_ms = int((frame_num - 1) / fps * 1000)
            entity = AnnoEntity(time=time_ms, id=obj_id)
            entity.bbox = [x, y, w, h]
            blob = {
                "frame_csv": frame_num,
                "frame_idx": frame_num - 1,
                "visibility": visibility
            }
            entity.labels = {}
            entity.labels["vis"] = visibility
            entity.labels[label_text] = label

            entity.confidence = conf
            entity.blob = blob

            sample.add_entity(entity)
    return sample


def main(args, description="Initial ingestion"):
    dataset_path = args.dataset_path
    out_filename = args.anno_name

    out_dataset = GluonCVMotionDataset(out_filename, dataset_path, load_anno=False)
    metadata = {
        FieldNames.DESCRIPTION: description,
        FieldNames.DATE_MODIFIED: str(datetime.datetime.now()),
    }
    out_dataset.metadata = metadata

    splits = {
        "test": os.path.join(out_dataset.data_root_path, "test"),
    }

    
    for split_name, split_path in splits.items():
        subdirs = glob.glob(os.path.join(split_path, "*"))
        for i, subdir in enumerate(subdirs):
            vid_id = os.path.basename(subdir)
            vid_path = os.path.join(split_path, vid_id)

            sample = DataSample(vid_id)

            info_path = os.path.join(vid_path, "seqinfo.ini")
            config = configparser.ConfigParser()
            config.read(info_path)
            seq_conf = config['Sequence']
            fps = float(seq_conf['frameRate'])
            num_frames = int(seq_conf['seqLength'])
            width = int(seq_conf['imWidth'])
            height = int(seq_conf['imHeight'])

            rel_base_dir = vid_path.replace(out_dataset.data_root_path, "").lstrip(os.path.sep)
            rel_base_dir = os.path.join(rel_base_dir, "img1")
            metadata = {
                FieldNames.DATA_PATH: rel_base_dir,
                FieldNames.FPS: fps,
                FieldNames.NUM_FRAMES: num_frames,
                FieldNames.RESOLUTION: {"width": width, "height": height},
            }
            sample.metadata = metadata

            # gt_path = os.path.join(vid_path, "gt/gt.txt")
            # has_gt = os.path.exists(gt_path)
            # anno_path = gt_path

            gt_path = os.path.join(vid_path, "gt/gt.txt")
            det_path = os.path.join(vid_path, "det/det.txt")
            has_gt = False
            anno_path = gt_path if has_gt else det_path

            sample = sample_from_mot_csv(anno_path, fps, sample, has_gt)

            out_dataset.add_sample(sample)

            print("Done {} sample {}/{}, {}".format(split_name, i+1, len(subdirs), vid_id))

    out_dataset.dump()

    return out_dataset


def write_data_split(args, dataset):
    if dataset is None:
        dataset = GluonCVMotionDataset(args.anno_name, args.dataset_path)

    def split_func(sample):
        data_path = sample.data_relative_path
        if data_path.startswith("train"):
            return SplitNames.TRAIN
        elif data_path.startswith("test"):
            return SplitNames.TEST

        raise Exception("Shouldn't happen")

    process_dataset_splits(dataset, split_func, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest mot dataset')
    parser.add_argument('--dataset_path', default="",
                        help="The path of dataset folder")
    parser.add_argument('--anno_name', default="anno.json",
                        help="The file name (with json) of ingested annotation file")
    args = parser.parse_args()
    dataset = main(args)
    write_data_split(args, dataset)
