# -*- coding: utf-8 -*-
# @Time : 2021/08/06 14:31
# @Author : yunshan
# @File : make_object_points.py
import numpy as np


def make_object_points(image_number):
    x_list = []
    for x in range(0,7):
        x_list.append(x*2.5)

    y_list = x_list.copy()

    object_points = []
    for i in range(image_number):
        for x in x_list:
            for y in y_list:
                object_points.append([x,y,0.0])

    return np.array(object_points)

if __name__ == '__main__':
    object_positions = make_object_points(image_number=44)

    with open('object_points.txt','w',encoding='utf-8') as file:
        for i in range(len(object_positions)):
            file.write('{} {} {}\n'.format(object_positions[i][0],object_positions[i][1],object_positions[i][2]))


