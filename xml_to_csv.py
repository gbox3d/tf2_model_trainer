import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
# import sys
import argparse


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            print(value)
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():

    parser = argparse.ArgumentParser(description="xml to csv")
    parser.add_argument('-d', '--directory', type=str, required=True, help='data set directory')

    args = parser.parse_args()
    base_path = args.directory

    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), (f'{base_path}/{folder}'))
        print(f'convert {image_path}')
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv((f'{base_path}/{folder}_labels.csv'), index=None)
    print('Successfully converted xml to csv.')

# # print(sys.argv)
# if len(sys.argv) >= 2 :
#     base_path = sys.argv[1]
#     print(f'base path {base_path}')


main()