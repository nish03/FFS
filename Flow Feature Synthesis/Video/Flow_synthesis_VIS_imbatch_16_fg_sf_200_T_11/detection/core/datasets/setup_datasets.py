import os
from collections import ChainMap
# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Project imports
import core.datasets.metadata as metadata
from pycocotools.coco import COCO

def setup_all_datasets(dataset_dir, image_root_corruption_prefix=None):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    setup_voc_dataset(dataset_dir)
    setup_coco_dataset(
        dataset_dir,
        image_root_corruption_prefix=image_root_corruption_prefix)
    setup_coco_ood_dataset(dataset_dir)
    setup_openim_odd_dataset(dataset_dir)
    setup_bdd_dataset(dataset_dir)
    setup_coco_ood_bdd_dataset(dataset_dir)
    register_vis_dataset()
    register_all_bdd_tracking(dataset_dir)
    register_all_coco(dataset_dir)
    register_coco_ood_wrt_bdd(dataset_dir)

def setup_coco_dataset(dataset_dir, image_root_corruption_prefix=None):
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    train_image_dir = os.path.join(dataset_dir, 'train2017')

    if image_root_corruption_prefix is not None:
        test_image_dir = os.path.join(
            dataset_dir, 'val2017' + image_root_corruption_prefix)
    else:
        test_image_dir = os.path.join(dataset_dir, 'val2017')

    train_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017.json')

    register_coco_instances(
        "coco_2017_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "coco_2017_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_openim_dataset(dataset_dir):
    """
    sets up openimages dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    # import ipdb; ipdb.set_trace()
    test_image_dir = os.path.join(dataset_dir, 'images')

    test_json_annotations = os.path.join(
        dataset_dir, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_val").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_openim_odd_dataset(dataset_dir):
    """
    sets up openimages out-of-distribution dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    test_image_dir = os.path.join(dataset_dir + 'ood_classes_rm_overlap', 'images')

    test_json_annotations = os.path.join(
        dataset_dir + 'ood_classes_rm_overlap', 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_ood_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_ood_val").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID



def setup_voc_id_dataset(dataset_dir):
    train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    train_json_annotations = os.path.join(
        dataset_dir, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train_id",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train_id").thing_classes = metadata.VOC_ID_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train_id").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain

    register_coco_instances(
        "voc_custom_val_id",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val_id").thing_classes = metadata.VOC_ID_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val_id").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain



def setup_bdd_dataset(dataset_dir):
    train_image_dir = os.path.join(dataset_dir, 'images/100k/train')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'images/100k/val')

    train_json_annotations = os.path.join(
        dataset_dir, 'train_bdd_converted.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_bdd_converted.json')

    register_coco_instances(
        "bdd_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "bdd_custom_train").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "bdd_custom_train").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "bdd_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "bdd_custom_val").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "bdd_custom_val").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_voc_dataset(dataset_dir):
    train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    train_json_annotations = os.path.join(
        dataset_dir, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train").thing_classes = metadata.VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "voc_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val").thing_classes = metadata.VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_voc_ood_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_ood_val").thing_classes = metadata.VOC_OOD_THING_CLASSES
    MetadataCatalog.get(
        "voc_ood_val").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain


def setup_coco_ood_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'val2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017_ood_rm_overlap.json')

    register_coco_instances(
        "coco_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_coco_ood_bdd_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'val2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017_ood_wrt_bdd_rm_overlap.json')

    register_coco_instances(
        "coco_ood_val_bdd",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_val_bdd").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_val_bdd").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_coco_ood_train_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'train2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017_ood.json')

    register_coco_instances(
        "coco_ood_train",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openimages_ood_oe_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'images')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_ood_oe",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "openimages_ood_oe").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_ood_oe").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID
        
######################################################################################
######################################################################################
##### Start: Code for custom registration of coco instances
######################################################################################
######################################################################################
  
"""Customized COCO data loader with additional keys."""
import numpy as np
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks

import contextlib
import datetime
import io
import json
import logging
import os
import shutil

from detectron2.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)

__all__ = ["load_coco_json", "register_custom_coco_instances"]


def load_coco_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(
                json_file, timer.seconds()
            )
        )

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [
            c["name"] for c in sorted(cats, key=lambda x: x["id"])
        ]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [
            ann["id"] for anns_per_image in anns for ann in anns_per_image
        ]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique!".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(
            len(imgs_anns), json_file
        )
    )

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (
        extra_annotation_keys or []
    )

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        record["ignore"] = img_dict["ignore"] if "ignore" in img_dict else 0
        record["video_id"] = (
            img_dict["video_id"] if "video_id" in img_dict else -1
        )
        record["index"] = (
            img_dict["frame_id"] if "frame_id" in img_dict else -1
        )

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id
            if anno.get("ignore", 0) != 0:
                continue
            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    # breakpoint()
                    segm = [
                        poly
                        for poly in segm
                        if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_custom_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).
    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        **metadata,
    ) 


####################################################################
########### Start registration of custom coco instances#############
####################################################################

  
def register_vis_dataset():
    thing_classes = ['airplane', 'bear', 'bird', 'boat', 'car', 'cat', 'cow', 'deer', 'dog', 'duck',
                   'earless_seal', 'elephant', 'fish', 'flying_disc', 'fox', 'frog', 'giant_panda',
                   'giraffe', 'horse', 'leopard', 'lizard', 'monkey', 'motorbike', 'mouse', 'parrot',
                   'person', 'rabbit', 'shark', 'skateboard', 'snake', 'snowboard', 'squirrel', 'surfboard',
                   'tennis_racket', 'tiger', 'train', 'truck', 'turtle', 'whale', 'zebra']
        # thing_classes = ['airplane', 'bear', 'car', 'cat', 'deer', 'dog','elephant',  'giant_panda',
        #                      'giraffe', 'horse', 'motorbike',
        #                      'person', 'rabbit', 'skateboard', 'snake',
        #                      'tiger','truck' ]
    VIS_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(ChainMap(*[{i + 1: i} for i in range(40)]))
    metadata = {"thing_classes": thing_classes}
    
    register_coco_instances(
            'vis21_val',
            metadata,
            '/projects/p084/p_discoret/Youtube-VIS/train/instances_val.json',
            "/projects/p084/p_discoret/Youtube-VIS/train/JPEGImages/",
    )
    register_coco_instances(
            'vis21_train',
            metadata,
            '/projects/p084/p_discoret/Youtube-VIS/train/instances_train.json',
            "/projects/p084/p_discoret/Youtube-VIS/train/JPEGImages/",
    )  
    MetadataCatalog.get(
        "vis21_train").thing_dataset_id_to_contiguous_id = VIS_THING_DATASET_ID_TO_CONTIGUOUS_ID
    MetadataCatalog.get(
        "vis21_val").thing_dataset_id_to_contiguous_id = VIS_THING_DATASET_ID_TO_CONTIGUOUS_ID
    
    
def register_all_bdd_tracking(dataset_dir):
    # BDD100K MOT set domain splits.
    _PREDEFINED_SPLITS_BDDT = {
        "bdd_tracking_2k": {
            "bdd_tracking_2k_train": (
                "/projects/p084/p_discoret/BDD100k_video/bdd100k/images/track/train",
                "/projects/p084/p_discoret/BDD100k_video/bdd100k/labels/track/bdd100k_mot_train_coco.json",
            ),
            "bdd_tracking_2k_val": (
                "/projects/p084/p_discoret/BDD100k_video/bdd100k/images/track/val",
                "/projects/p084/p_discoret/BDD100k_video/bdd100k/labels/track/bdd100k_mot_val_coco.json",
            ),
        },
    }
    # bdd_tracking meta data
    # fmt: off
    thing_classes = ['pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle']
    # thing_classes = ['person', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
    thing_classes_3cls = ["vehicle", "pedestrian", "cyclist"]
    # fmt: on
    for DATASETS in [_PREDEFINED_SPLITS_BDDT]:
        for key, value in DATASETS.items():
            metadata = {
                "thing_classes": thing_classes_3cls
                if "3cls" in key
                else thing_classes
            }
            for name, (img_dir, label_file) in value.items():
                register_coco_instances(
                    name,
                    metadata,
                    os.path.join(dataset_dir, label_file),
                    os.path.join(dataset_dir, img_dir),
                )

    #register the nuscene ood dataset.
    metadata = {  "thing_classes": ['car', 'truck',
                                        'trailer', 'bus',
                                        'construction_vehicle', 'bicycle',
                                        'motorcycle', 'pedestrian',
                                        'traffic_cone', 'barrier', 'animal', 'debris',
                                        'pushable_pullable', 'mobility', 'stroller', 'wheelchair']}
    register_coco_instances(
        'nu_bdd_ood',
        metadata,
        '/projects/p084/p_discoret/nuImages/nu_ood.json',
        "/projects/p084/p_discoret/nuImages/",
    )
    coco_api = COCO('/projects/p084/p_discoret/nuImages/nu_ood.json')
    cat_ids = sorted(coco_api.getCatIds())
    id_map = {v: i for i, v in enumerate(cat_ids)}
    MetadataCatalog.get("nu_bdd_ood").thing_dataset_id_to_contiguous_id = id_map
    # register the coco ood dataset wrt the vis dataset.
    metadata = {
        "thing_classes": ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                              'bus', 'train', 'truck', 'boat', 'traffic light',
                              'fire hydrant', 'stop sign', 'parking meter', 'bench',
                              'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                              'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                              'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                              'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                              'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                              'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                              'hair drier', 'toothbrush']
    }
    register_coco_instances(
        'vis_coco_ood',
        metadata,
        '/projects/p084/p_discoret/COCO/annotations/instances_val2017_ood_wrt_vis.json',
        "/projects/p084/p_discoret/COCO/train2017/",
    )    
    coco_api = COCO('/projects/p084/p_discoret/COCO/annotations/instances_val2017_ood_wrt_vis.json')
    cat_ids = sorted(coco_api.getCatIds())
    id_map = {v: i for i, v in enumerate(cat_ids)}
    MetadataCatalog.get("vis_coco_ood").thing_dataset_id_to_contiguous_id = id_map
    
    
    
    
    
def register_all_coco(dataset_dir):
    thing_classes = MetadataCatalog.get('coco_2017_train').thing_classes
    metadata = {"thing_classes": thing_classes}
    train_json_annotations = os.path.join(
            dataset_dir, 'annotations', 'instances_train2017.json')
    test_json_annotations = os.path.join(
            dataset_dir, 'annotations', 'instances_val2017.json')
    train_image_dir = os.path.join(dataset_dir, 'train2017')
    test_image_dir = os.path.join(dataset_dir, 'val2017')
    register_coco_instances('coco_2017_train_custom', metadata,
                                train_json_annotations,
                                train_image_dir)
    register_coco_instances('coco_2017_val_custom', metadata,
                                test_json_annotations,
                                test_image_dir)
                                
                                
                                

def register_coco_ood_wrt_bdd(dataset_dir):
    thing_classes = MetadataCatalog.get('coco_2017_train').thing_classes
    metadata = {"thing_classes": thing_classes}

    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017_ood_wrt_bdd.json')
    test_image_dir = os.path.join(dataset_dir, 'val2017')

    register_coco_instances('coco_2017_val_ood_wrt_bdd', metadata,
                                test_json_annotations,
                                  test_image_dir)
                                  
                                  
                                  
                                  
######################################################################################
######################################################################################
##### End: Code for custom registration of coco instances
######################################################################################
######################################################################################