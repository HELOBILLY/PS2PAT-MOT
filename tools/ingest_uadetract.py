# Ingestion script for UA-DETRAC (https://detrac-db.rit.albany.edu/) dataset.

import os
import sys
import tqdm
import argparse
import pathlib

from xml.etree import ElementTree
from datetime import datetime

import gluoncv.torch.data.gluoncv_motion_dataset.dataset as gcvdataset
gcvdataset.__version__ = "1.0"

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import (
    AnnoEntity, DataSample, FieldNames, GluonCVMotionDataset, SplitNames
)
from gluoncv.torch.data.gluoncv_motion_dataset.utils.ingestion_utils import \
    process_dataset_splits


# _VEHICLE_TYPES = (
#     'Bus', 'Hatchback', 'MiniVan', 'Police', 'Sedan', 'Suv', 'Taxi',
#     'Truck-Box-Large', 'Truck-Box-Med', 'Truck-Flatbed', 'Truck-Pickup',
#     'Truck-Util', 'Van'
# )

_VEHICLE_TYPES = ('car', 'van', 'bus', 'others')

_CLASS_LABELS = dict((vt, i) for i, vt in enumerate(_VEHICLE_TYPES, start=1))


def sample_from_xml(xml_file_path, split_dir_name, args):
    def _read_box(node_attr):
        def _coord(x):
            return round(float(x))
        
        x = _coord(node_attr['left'])
        y = _coord(node_attr['top'])
        w = _coord(node_attr['width'])
        h = _coord(node_attr['height'])

        return [x, y, w, h]

    tree = ElementTree.parse(xml_file_path)
    root = tree.getroot()

    seq_name = root.attrib['name']
    sample = DataSample(id=seq_name)

    frame_num = 0
    for frame in root.findall('./frame'):
        frame_num = int(frame.attrib['num'])
        frame_idx = frame_num - 1

        for target in frame.findall('.//target'):
            obj_id = int(target.attrib['id'])
            time_ms = int(round((frame_idx / args.fps) * 1000))

            entity = AnnoEntity(time=time_ms, id=obj_id)
            entity.confidence = 1.0

            box_attr = target.find('box').attrib
            entity.bbox = _read_box(box_attr)
                        
            attrib_attr = target.find('attribute').attrib
            vehicle_type = attrib_attr['vehicle_type']
            entity.blob = {
                'frame_xml':         frame_num,
                'frame_idx':         frame_idx,
                # 'color':             attrib_attr['color'],
                'orientation':       float(attrib_attr['orientation']),
                'speed':             float(attrib_attr['speed']),
                'trajectory_length': float(attrib_attr['trajectory_length']),
                'truncation_ratio':  float(attrib_attr['truncation_ratio']),
                'vehicle_type':      vehicle_type,
            }
            entity.labels = {vehicle_type: _CLASS_LABELS[vehicle_type]}
            
            region_overlap = target.find('.//region_overlap')
            if region_overlap is not None:
                region_overlap_attr = region_overlap.attrib
                occlusion_status = region_overlap_attr['occlusion_status']
                occlusion_box = _read_box(region_overlap_attr)
                entity.blob['occlusion_status'] = int(occlusion_status)
                entity.blob['occlusion_box'] = occlusion_box

            sample.add_entity(entity)
    
    # Need to replace the Windows path separator by UNIX-like to make the path
    # working across different platforms. Linux struggles with mixing path
    # separators whereas Windows does not.
    rel_data_path = os.path.join(split_dir_name, seq_name).replace('\\', '/')
    sample.metadata = {
        FieldNames.DATA_PATH:  rel_data_path,
        FieldNames.FPS:        args.fps,
        FieldNames.NUM_FRAMES: frame_num,
        FieldNames.RESOLUTION: {
            'width': args.img_width, 'height': args.img_height
        }
    }

    return sample


def ingest_uadetrac(args):
    dataset = GluonCVMotionDataset(
        annotation_file='anno.json', root_path=args.dataset_dir_path,
        load_anno=False
    )
    dataset.metadata = {
        FieldNames.DESCRIPTION:   "UA-DETRAC benchmark dataset XML ingestion",
        FieldNames.DATE_MODIFIED: str(datetime.now()),
    }

    dataset_anno_dir = (
        pathlib.Path(args.dataset_dir_path) / GluonCVMotionDataset.DATA_DIR /
        'DETRAC_public'
    )
    splits = (
        ('Insight-MVT_Annotation_Train', dataset_anno_dir / '540p-Train'),
        ('Insight-MVT_Annotation_Test', dataset_anno_dir / '540p-Test'),
    )

    tqdm_pbar = tqdm.tqdm(file=sys.stdout)
    with tqdm_pbar as pbar:
        for split_dir_name, split_dir in splits:
            # print('split_dir_name=======>', split_dir_name)
            # print('split_dir=======>', split_dir)

            for sample_xml_file_path in map(str, split_dir.iterdir()):
                pbar.set_description(f"reading sample {sample_xml_file_path}")
                sample = sample_from_xml(
                    sample_xml_file_path, split_dir_name, args
                )
                dataset.add_sample(sample)
                pbar.update()
    
    dataset.dump()

    return dataset


def write_data_split(dataset):
    def split_func(sample):
        data_path = sample.data_relative_path

        if '_Train' in data_path:
            return SplitNames.TRAIN
        elif '_Test' in data_path:
            return SplitNames.TEST
        
        raise RuntimeError("unrecognized data split")
    
    process_dataset_splits(dataset, split_func, save=True)


def main():
    parser = argparse.ArgumentParser(
        description="UA-DETRAC dataset XML ingestion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dataset_dir_path', type=str,
        help="Root directory path to the dataset."
    )
    parser.add_argument(
        '--fps', type=int, default=25, help="FPS for all data samples."
    )
    parser.add_argument(
        '--img-width', type=int, default=960, help="Image width."
    )
    parser.add_argument(
        '--img-height', type=int, default=540, help="Image height"
    )
    args = parser.parse_args()

    dataset = ingest_uadetrac(args)
    write_data_split(dataset)

    return 0


if __name__ == '__main__':
    sys.exit(main())
