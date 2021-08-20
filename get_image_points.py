# -*- coding: utf-8 -*-
# @Time : 2021/08/19 16:39
# @Author : yunshan
# @File : get_image_points.py
import cv2

from sort_corners_box import Sort_corners_box

image_points = Sort_corners_box(w=7,h=7)

with open('image_points_left.txt', 'w') as file :
    for index in range(44):
        image = cv2.imread('./left_img/{}.png'.format(index))
        result_image,corners_box = image_points.run(image,flag=0)

        for corner in corners_box:
            file.write('{},{}\n'.format(corner[0],corner[1]))

        print(f"index:{index}")

        cv2.namedWindow("result_image", 0)
        cv2.imshow("result_image", result_image)
        cv2.waitKey(200)

