from math import floor
import pandas as pd
import os

from datasetinsights.datasets.unity_perception import AnnotationDefinitions, MetricDefinitions
from datasetinsights.datasets.unity_perception.captures import Captures

from PIL import Image

class FileFormatError(Exception):
    pass

def compute_yolo_param(x_abs_raw: int, y_abs_raw: int, width_abs_raw: int,
                    height_abs_raw: int, image_width: int,
                    image_height: int) -> list:
    """The function calculates labels box in YoLo format

    Args:
        x_abs_raw (int): top-left x coordinates
        y_abs_raw (int): top-left y coordinates
        width_abs_raw (int): width label box in absalute coordinate
        height_abs_raw (int): height label box in absalute coordinate
        image_width (int): image width
        image_height (int): image height

    Returns:
        list: Yolo label box params <x_center> <y_center> <width> <height>
    """

    assert image_width != 0 and image_height != 0, "The width or length of the image is zero"

    # the center of the image in absolute coordinates
    x_abs_prep = x_abs_raw + floor(width_abs_raw/2)
    y_abs_prep = y_abs_raw + floor(height_abs_raw/2)

    # the center of the image in relative coordinates
    x_rel_prep = x_abs_prep / image_width
    y_rel_prep = y_abs_prep / image_height
    
    # width and height in relative coordinates
    width_rel_prep = width_abs_raw/ image_width
    height_rel_prep = height_abs_raw / image_height

    return [x_rel_prep, y_rel_prep, width_rel_prep, height_rel_prep]


def convent_to_yolo_format(raw_labels: list, image_size:tuple)->list:
    """The function generates files for labels in Yolo format

    Args:
        raw_labels (list): list of raw labels boxes
        image_params (tuple): image size (width, height)

    Returns:
        list: list of yolo labels <object-class> <x_center> <y_center> <width> <height>
    """
    labels = []
    image_width, image_height = image_size
    for raw_label in raw_labels:
        label_id = raw_label.get("label_id")
        x_abs_raw = raw_label.get('x')
        y_abs_raw = raw_label.get('y')
        width_abs_raw = raw_label.get('width')
        height_abs_raw = raw_label.get('height')
        labels.append([label_id] + compute_yolo_param(x_abs_raw, y_abs_raw, width_abs_raw, height_abs_raw, image_width, image_height))

    return labels

def save_to_file(content: list, file_name: str, path_to_save_dir: str) -> bool:
    """The function saves labels to the file

    Args:
        content (list): labels to be recorded
        file_name (str): name for the file
        path_to_save_dir (str): the path to the directory where the file will be saved

    Returns:
        bool: true if successfully save, else false
    """
    try:
        with open(os.path.join(path_to_save_dir, file_name+".txt"),"w") as txt_file:
            for line in content:
                txt_file.writelines(line)
    except FileNotFoundError:
        print("It is impossible to save the file, the specified directory does not exist")
        return False
    except:
        return False
    return True

def save_to_file_labels_name(labels_names: list, file_name: str,path_to_save_dir: str) -> bool:
    """The function saves labels name to the file

    Args:
        labels_names (list): list of labels name
        file_name (str): name for the file
        path_to_save_dir (str): the path to the directory where the file will be saved

    Returns:
        bool: true if successfully save, else false
    """
    try:
        with open(os.path.join(path_to_save_dir, file_name+".txt"),"w") as txt_file:
            for name in labels_names:
                txt_file.writelines(name)
    except FileNotFoundError:
        print("It is impossible to save the file, the specified directory does not exist")
        return False
    except:
        return False
    return True

def prepare_ds_info(base_dataset_dir: str, auto_mode = True, manual_img_size = (0,0)) -> tuple[pd.DataFrame, list]:
    """The functions prepare all information about dataset 

    Args:
        base_dataset_dir (str): current base dataset di
        auto_mode (bool): if true auto get image size mode else use manual image size from manual_img_size
        manual_img_size (tuple): manual image size, use if auto_mode False

    Returns:
        tuple[pd.DataFrame, list]: datsset info: image filenames, labels, labels name
    """

    assert  os.path.isdir(base_dataset_dir), "Not found base dataset dir"

    labels_info = []
    # get the parameters of the unity dataset using datasetinsights
    captures = Captures(base_dataset_dir).filter(def_id=AnnotationDefinitions(base_dataset_dir).table.to_dict('records')[0]["id"])
    # get image sizes auto or manual
    if auto_mode:
        # get the size for each image in a folder
        image_params = []
        for fn in captures["filename"]:
            temp = tuple()
            temp = Image.open(os.path.join(base_dataset_dir,fn)).size
            image_params.append(temp)
        pd_img_sizes = pd.Series(image_params).rename("img_params")
    else:
        # get sizes from manual_img_size
        image_params = []
        for _ in range(len(captures["filename"])):
            image_params.append(manual_img_size)
        pd_img_sizes = pd.Series(image_params).rename("img_params")
    captures = pd.concat([captures["filename"], captures["annotation.values"], pd_img_sizes], axis=1)

    # get the names of the labels
    annotation_def = AnnotationDefinitions(data_root=base_dataset_dir)
    definition_dict = annotation_def.get_definition(def_id=AnnotationDefinitions(base_dataset_dir).table.to_dict('records')[0]["id"])
    for lb in definition_dict['spec']:
        labels_info.append(lb["label_name"])

    return (captures, labels_info)

def convert(ds_info: tuple[pd.DataFrame, list], path_to_save_dir:str) -> bool:
    """The function takes input information about the dataset and generates labels in Yolo format

    Args:
        df_info tuple[pd.DataFrame, list]: dataset info image filenames, labels, labels name
        path_to_save_dir (str): path dir whare save yolo lables

    Returns:
        bool: true if successfully, else false
    """
    pd_df = ds_info[0]
    labels_name = ds_info[1]

    if not save_to_file_labels_name(labels_name, "object_names", path_to_save_dir):
        return False

    for _, row in pd_df.iterrows():
        row["annotation.values"] = convent_to_yolo_format(row["annotation.values"],
        row["img_params"])
        file_name = row["filename"].split("/")[1].split(".")[0]
        content = [" ".join(map(str,l))+"\n" for l in row["annotation.values"]]
        if not save_to_file(content, file_name, path_to_save_dir):
            return False

    return True