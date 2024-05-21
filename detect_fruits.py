import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm
import numpy as np
import imutils


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.
    Parameters
    ----------
    img_path : str
        Path to processed image.
    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # TODO: Implement detection method.

    img = imutils.resize(img, width=640)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # apple
    lowerA = np.array([0, 131, 41])
    upperA = np.array([10, 255, 255])

    maskA = cv2.inRange(hsv, lowerA, upperA)
    erodeA = cv2.erode(maskA, np.ones((11,11), np.uint8))
    threshA = cv2.threshold(erodeA, 0, 255, cv2.THRESH_OTSU)[1]
    morA = cv2.morphologyEx(threshA, cv2.MORPH_OPEN, np.ones((17,17), np.uint8))
    ret, imgtA = cv2.threshold(cv2.distanceTransform(morA, cv2.DIST_L2, 5),
                               0.5 * (cv2.distanceTransform(morA, cv2.DIST_L2, 5)).max(), 255, 0)
    konturyA = cv2.findContours(np.uint8(imgtA).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("apples:", len(imutils.grab_contours(konturyA)))

    # bananas
    lowerB = np.array([21, 51, 155])
    upperB = np.array([25, 255, 255])

    maskB = cv2.inRange(hsv, lowerB, upperB)
    erodeB = cv2.erode(maskB, np.ones((11,11), np.uint8))
    threshB = cv2.threshold(erodeB, 0, 255, cv2.THRESH_OTSU)[1]
    morB = cv2.morphologyEx(threshB, cv2.MORPH_OPEN, np.ones((17,17), np.uint8))
    ret, imgtB = cv2.threshold(cv2.distanceTransform(morB, cv2.DIST_L2, 5),
                               0.5 * (cv2.distanceTransform(morB, cv2.DIST_L2, 5)).max(), 255, 0)
    konturyB = cv2.findContours(np.uint8(imgtB).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("bananas:", len(imutils.grab_contours(konturyB)))


    # oranges
    lowerO = np.array([11, 50, 105])
    upperO = np.array([13, 255, 255])

    maskO = cv2.inRange(hsv, lowerO, upperO)
    erodeO = cv2.erode(maskO, np.ones((3,3), np.uint8))
    threshO = cv2.threshold(erodeO, 0, 255, cv2.THRESH_OTSU)[1]
    morO = cv2.morphologyEx(threshO, cv2.MORPH_OPEN, np.ones((17, 17), np.uint8))
    ret, imgtO = cv2.threshold(cv2.distanceTransform(morO, cv2.DIST_L2, 5),
                               0.5 * (cv2.distanceTransform(morO, cv2.DIST_L2, 5)).max(), 255, 0)
    konturyO = cv2.findContours((np.uint8(imgtO)).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("oranges:", len(imutils.grab_contours(konturyO)))



    return {'apple': len(imutils.grab_contours(konturyA)), 'banana': len(imutils.grab_contours(konturyB)), 'orange': len(imutils.grab_contours(konturyO))}


@click.command()

@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits



    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()