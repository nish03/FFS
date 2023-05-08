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
        



####################################################################
########### Start registration of video instances#############
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
                "/projects/p084/p_discoret/BDD100k_video/bdd100k/labels/track/bdd100k_det_train_coco.json",
            ),
            "bdd_tracking_2k_val": (
                "/projects/p084/p_discoret/BDD100k_video/bdd100k/images/track/val",
                "/projects/p084/p_discoret/BDD100k_video/bdd100k/labels/track/bdd100k_det_val_coco.json",
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
    BDDVID_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(ChainMap(*[{i + 1: i} for i in range(8)]))
    MetadataCatalog.get(
        "bdd_tracking_2k_train").thing_dataset_id_to_contiguous_id = BDDVID_THING_DATASET_ID_TO_CONTIGUOUS_ID
    MetadataCatalog.get(
        "bdd_tracking_2k_val").thing_dataset_id_to_contiguous_id = BDDVID_THING_DATASET_ID_TO_CONTIGUOUS_ID
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
    
    # # register the coco ood dataset wrt the vis dataset.
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
    # register the coco ood dataset wrt the vis dataset.  
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
    #coco_api = COCO('/projects/p084/p_discoret/COCO/annotations/instances_val2017_ood_wrt_bdd.json')
    #cat_ids = sorted(coco_api.getCatIds())
    #id_map = {v: i for i, v in enumerate(cat_ids)}
    MetadataCatalog.get("coco_2017_val_ood_wrt_bdd").thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
    'coco_2017_train').thing_dataset_id_to_contiguous_id                                  
                                  
                                  
                                  
######################################################################################
######################################################################################
##### End: Code for custom registration of video instances
######################################################################################
######################################################################################