from math import floor
import pandas as pd

class FileFormatError(Exception):
    pass

def compute_yolo_param(x_abs_raw: int, y_abs_raw: int, width_abs_raw: int,
                    height_abs_raw: int, image_width: int,
                    image_height: int) -> list:
    """
    Функция получает на вход коробку: координаты верхнего угла (x_abs_raw,y_abs_raw) и ширину,высоту (width_abs_raw,height_abs_raw),
    а также размер изображения (image_width, image_height). И возвращает коробку в формте для сети YoLo - лист из 4 чисел
    """


    # центр изображения в абсолютных координатах
    x_abs_prep = x_abs_raw + floor(width_abs_raw/2)
    y_abs_prep = y_abs_raw + floor(height_abs_raw/2)
    
    try:
        # центр изображения в относительных координатах
        x_rel_prep = x_abs_prep / image_width
        y_rel_prep = y_abs_prep / image_height
        
        # ширина и высота в относительных координатах
        width_rel_prep = width_abs_raw/ image_width
        height_rel_prep = height_abs_raw / image_height

    except ZeroDivisionError:
        print("The width or length of the image is zero")

    return [x_rel_prep, y_rel_prep, width_rel_prep, height_rel_prep]


def convent_to_yolo_format(raw_labels: list, image_params: dict)->list:
    """
    Функция формирует лейблы в формате Yolo:
    <object-class> <x_center> <y_center> <width> <height>
    """
    labels = []
    try:
        image_width = image_params.get("width")
        image_height = image_params.get("height")
        for raw_label in raw_labels:
            label_id = raw_label.get("label_id")
            x_abs_raw = raw_label.get('x')
            y_abs_raw = raw_label.get('y')
            width_abs_raw = raw_label.get('width')
            height_abs_raw = raw_label.get('height')
            labels.append([label_id] + compute_yolo_param(x_abs_raw, y_abs_raw, width_abs_raw, height_abs_raw, image_width, image_height))
    except KeyError:
        print("Raw labels has an incorrect structure")
    return labels

def save_to_file(content: list, file_name: str, path_to_save_dir: str):
    """
    Функция сохраняет строки в content в файл path_to_save_dir/file_name.txt
    """
    try:
        with open(path_to_save_dir+file_name+".txt","w") as txt_file:
            for line in content:
                txt_file.writelines(line)
    except FileNotFoundError:
        print("It is impossible to save the file, the specified directory does not exist")
        return False
    except:
        return False
    return True


def convent(pd_df: pd.DataFrame, path_to_save_dir:str):
    """
    Функция принимает на вход информацию о датасете(image filenames,labels,image params)
    и формирует лайеблы в формате Yolo, сохроняя их в path_to_save_dir
    """
    #pd_labels = pd_labels.apply(convent_to_yolo_format)
    #pd_df = pd.concat([pd_filenames, pd_labels, pd_img_params ], axis=1)
    for _, row in pd_df.iterrows():
        row["annotation.values"] = convent_to_yolo_format(row["annotation.values"],
        row["img_params"])
        file_name = row["filename"].split("/")[1].split(".")[0]
        #file_name = file_name.split(".")[0]
        content = [" ".join(map(str,l))+"\n" for l in row["annotation.values"]]
        if save_to_file(content, file_name, path_to_save_dir) == 0:
            return False
    return True