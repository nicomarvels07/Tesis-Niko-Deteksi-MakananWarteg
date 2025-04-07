import argparse
import os
from defusedxml import ElementTree as ET
from typing import Dict
from tqdm import tqdm

class XMLToTXTConverter:

    def __init__(self, annotations_path: str, labels_path: str, classes: Dict[str, int]):
        self.annotations_path: str = annotations_path
        self.labels_path: str = labels_path
        self.classes: Dict[str, int] = classes

    def convert_annotation(self, annotation_file: str) -> None:
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        filename: str = root.find('filename').text.replace('.jpg', '.txt')
        image_size = root.find('size')
        image_width: int = int(image_size.find('width').text)
        image_height: int = int(image_size.find('height').text)

        label_file: str = os.path.join(self.labels_path, filename)

        with open(label_file, 'w') as file:
            for obj in root.iter('object'):
                class_name: str = obj.find('name').text
                if class_name not in self.classes:
                    continue  # Skip if class not in provided class mapping
                class_id: int = self.classes[class_name]
                bndbox = obj.find('bndbox')
                xmin: int = int(bndbox.find('xmin').text)
                ymin: int = int(bndbox.find('ymin').text)
                xmax: int = int(bndbox.find('xmax').text)
                ymax: int = int(bndbox.find('ymax').text)

                x_center_norm: float = (xmin + xmax) / (2 * image_width)
                y_center_norm: float = (ymin + ymax) / (2 * image_height)
                width_norm: float = (xmax - xmin) / image_width
                height_norm: float = (ymax - ymin) / image_height

                file.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

    def convert_all(self) -> None:
        self.create_directory(self.labels_path)
       
        annotation_files = os.listdir(self.annotations_path)
        for annotation in tqdm(annotation_files, desc="Converting XML to TXT"):
            annotation_file: str = os.path.join(self.annotations_path, annotation)
            self.convert_annotation(annotation_file)

    @staticmethod
    def create_directory(directory_path: str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

def parse_classes(classes_str: str) -> Dict[str, int]:
    class_pairs = classes_str.split(',')
    return {pair.split('=')[0].strip(): int(pair.split('=')[1].strip()) for pair in class_pairs}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XML annotations to TXT format for object detection models.')

    parser.add_argument('--annotations_path', type=str, default='dataset_mentah/annotations', help='Directory path where the XML files are stored.')
    parser.add_argument('--labels_path', type=str, default='dataset_mentah/labels', help='Output directory path where the TXT files will be saved.')
    parser.add_argument('--classes', type=str, default='Nasi=0,Tempe=1,Sambel=2,Mie=3,Sayur_kol=4,Tempe_orek=5,Telur_ceplok=6,Telur_ceplok_sambel=7,Telur_bulat_sambel=8,Tumis_oncom_cepokak=9,Sayur_timun=10,Gorengan=11,Sambel_tempe=12,Sayur_bayam=13,Telur_dadar=14,Rawon=15,Kerang=16,Tumis_toge=17,Ikan_kembung=18,Sayur_daun_singkong=19,Ayam_sambel=20,Sambel_terong=21,Kikil=22,Tumis_tempe_cepokak=23,Sambel_kentang=24,Teri=25,Tumis_toge_kacang_panjang=26,Semur_jengkol=27,Capcai=28,Tumis_labu_wortel=29,Tumis_pare=30,Kwetiau=31,Tumis_kacang_panjang=32,Bergedel=33,Ikan_tongkol_sambel=34,Sayur_labu=35,Tumis_jamur=36,Sayur_sawi=37,Tumis_tempe_buncis=38,Sayur_nangka=39,Ikan_patin=40,Tumis_tempe=41,Tumis_kangkung_buncis=42,Ati_ampela=43,Ayam_kecap=44,Tumis_kangkung=45,Sayur_sawi_jagung=46,Sayur_bambu=47,Ikan_asin=48,Tumis_toge_genjer=49,Ikan_kembung_sambel=50,Tumis_jagung=51,Sayur_buncis=52,Usus=53,Sayur_tahu_kacang_panjang=54,Tumis_tahu=55,Tumis_oncom=56,Tumis_sawi=57,Ikan_patin_sambel=58,Sayur_kacang_panjang=59,Sayur_kentang=60,Ikan_patin_sayur=61,Rendang_daging=62,Kentang=63,Bihun=64,Telur_bulat=65,Tumis_buncis_wortel=66,Ayam_sayur_kentang=67,Tumis_kentang_buncis=68,Tumis_buncis=69,Tahu=70,Sayur_tempe_kentang=71,Sayur_kentang_wortel=72,Tumis_kangkung_tempe=73,Tumis_kol_bayam=74,Telur_sambel=75,Ayam_sayur=76,Sambel_tongkol_kentang=77,Telur_tempe_kecap=78,Pepes_tahu=79,Tumis_daun_pepaya=80,Tumis_bunga_pepaya=81,Tumis_tahu_terong=82,Tumis_jamur_buncis=83,Tumis_terong=84,Tumis_tempe_kacang_panjang=85,Sambel_ceker=86,Sayur_tempe=87,Sayur_tahu_toge=88,Kacang_panjang=89,Ikan_bawal=90,Tumis_toge_tahu=91,Daun_kangkung=92,Semur_telur_bulat_tempe=93,Tumis_kentang_kacang_panjang=94,Tumis_toge_wortel=95', help='Comma-separated list of class mappings in the format "class1=id1,class2=id2,...".')
    
    args = parser.parse_args()

    classes: Dict[str, int] = parse_classes(args.classes)

    # Create an instance of the converter and convert all XML files to TXT format
    converter = XMLToTXTConverter(args.annotations_path, args.labels_path, classes)
    converter.convert_all()
