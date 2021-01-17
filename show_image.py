from multiprocessing import Pool
import cv2
import numpy as np
import pandas as pd

def draw(input):
    image_path, wind_speed = input
    content = str(wind_speed)
    from_path = '/home/giang/Desktop/Wind_data/train/' + image_path + '.jpg'
    to_path = '/home/giang/Desktop/Wind_data/show_image/' + image_path + '.jpg'
    image = cv2.imread(from_path)
    image = cv2.putText(image, content, (50,50), cv2.FONT_HERSHEY_SIMPLEX,\
        1, [0,255,255], 2, cv2.LINE_AA)
    cv2.imwrite(to_path, image)

if __name__ == "__main__":
    df = pd.read_csv('/home/giang/Desktop/Wind_data/training_set_labels.csv')
    name = df.image_id.to_list()
    speed = df.wind_speed.to_list()
    with Pool(8) as p:
        p.map(draw, zip(name, speed))
    # draw(name[0], speed[0])