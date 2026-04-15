import pathlib
import xml.etree.ElementTree as ET
from PIL import Image
import shutil



class NewData:
    def __init__(self,path_floder_xml=None, path_floder_dataset=None):
        self.path_data = path_floder_dataset
        self.path_floder_xml = path_floder_xml


    def convert_from_xml_to_YOLO(self, name_label='annotations', name_images='images'):
        path_floder_label = pathlib.Path(f'{self.path_floder_xml}\\{name_label}')
        path_floder_imgs = pathlib.Path(f'{self.path_floder_xml}\\{name_images}')
        list_names_cls = list()
        for item in path_floder_label.iterdir():
            tree = ET.parse(str(item))
            # print(str(item.stem))
            root = tree.getroot()
            '''get w,h img'''
            image_path = path_floder_imgs / f'{item.stem}.png'
            if  not image_path.exists():
                print(f'изображение не найдено попути {image_path}')
                continue

            with Image.open(str(image_path)) as img:
                width, height = img.size
            # print(width, height)
            '''------------'''
            for obj in root.findall('object'):
                name = obj.find('name').text
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                print(name, xmin, ymin, xmax, ymax)
                if name not in list_names_cls:
                    list_names_cls.append(name)
                box_w = xmax - xmin
                box_h = ymax - ymin
                x_center = xmin + (box_w/2)
                y_center = ymin + (box_h/2)

                yolo_x = x_center/width
                yolo_y = y_center/height
                yolo_w = box_w/width
                yolo_h = box_h/height 

                new_floder_labels = pathlib.Path(f'{self.path_data}\\labels')              
                new_floder_images = pathlib.Path(f'{self.path_data}\\images')

                new_floder_images.mkdir(exist_ok=True, parents=True)
                new_floder_labels.mkdir(exist_ok=True, parents=True)
                
                txt_path = new_floder_labels / f'{item.stem}.txt'
                
                if txt_path.exists() and txt_path.stat().st_size >0:
                    continue
                else:
                    with txt_path.open('a', encoding='utf-8') as f:
                        f.write(f'{1} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}\n')
                    
            target_image_path = new_floder_images / f'{item.stem}.png'
            if not target_image_path.exists():
                shutil.move(str(image_path), str(target_image_path))
    
def unif_cls(path_labels_yolo, inp, out):
    path_floder = pathlib.Path(path_labels_yolo)
    for item in path_floder.iterdir():
        if item.suffix != '.txt':
            continue  # Обрабатываем только текстовые файлы
        lines = item.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if parts and parts[0] == str(inp):
                parts[0] = str(out)
            new_lines.append(' '.join(parts))
        item.write_text('\n'.join(new_lines)+ '\n')

def delete_cls(path_labels_yolo, cls_to_delete):
    """
    Удаляет строки с указанным классом из всех файлов в папке.
    cls_to_delete: строка или список строк, например '4' или ['3', '4']
    """
    path_folder = pathlib.Path(path_labels_yolo)
    
    if isinstance(cls_to_delete, str):
        cls_to_delete = [cls_to_delete]
    else:
        cls_to_delete = [str(c) for c in cls_to_delete]

    for item in path_folder.iterdir():
        if item.suffix != '.txt':
            continue
            
        lines = item.read_text().splitlines()
       
        new_lines = [line for line in lines if line.split() and line.split()[0] not in cls_to_delete]
        
        
        item.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''))





# adder = NewData(path_floder_xml='new_data' , path_floder_dataset='data_converted')
# adder.convert_from_xml_to_YOLO()
        
# unif_cls('RDD_SPLIT\\val\\labels','1', '0')

# unif_cls('RDD_SPLIT\\val\\labels','3', '1')
# unif_cls('RDD_SPLIT\\train\\labels','3', '2')


# delete_cls('RDD_SPLIT/val/labels', '4')





# import pathlib

def find_bad_labels(path_labels, max_idx=2):
    path = pathlib.Path(path_labels)
    bad_files = []
    for txt_file in path.glob('**/*.txt'):
        lines = txt_file.read_text().splitlines()
        for line in lines:
            parts = line.split()
            if parts and int(parts[0]) > max_idx:
                bad_files.append((txt_file, parts[0]))
                break
    return bad_files

print("Ошибки в train:", find_bad_labels('RDD_SPLIT/train/labels'))
print("Ошибки в train:", find_bad_labels('RDD_SPLIT/val/labels'))