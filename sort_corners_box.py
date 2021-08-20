# -*- coding: utf-8 -*-
# @Time : 2021/08/20 09:32
# @Author : yunshan
# @File : test.py

import time

import cv2
import numpy as np
import copy


class Sort_corners_box:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self._total_corners = self.w * self.h
        self._patternSize = (self.w, self.h)

    def _draw_line(self, image, p1, p2, line_color=(0, 0, 255)):
        cv2.line(
            image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), line_color, 5
        )

    def _calc_dis2points(self, point1, point2):
        """
        calculate distance of two points
        :param point1:list or tuple
        :param point2:list or tuple
        :return:distance
        """
        vector = np.array(point1) - np.array(point2)
        distance = round(np.linalg.norm(vector), 4)
        return distance

    def _distance_Point2Line(self, point_A, point_B, target_point):
        vector_A = np.array(target_point) - np.array(point_A)
        vector_B = np.array(point_B) - np.array(point_A)
        B_mo = np.linalg.norm(vector_B)
        AXB = np.cross(vector_A, vector_B)
        AXB_mo = np.linalg.norm(AXB)
        distance = AXB_mo / B_mo
        return distance

    def _vector_A2B(self, position_A, position_B):
        vevtor_A = np.array(position_A)
        vector_B = np.array(position_B)
        vector = vector_B - vevtor_A
        return vector

    def _get_corners_box(self, image):
        """
        :param image: image
        :return: corner box :type list
        """
        corners_box = []
        ret, corners = cv2.findCirclesGrid(
            image, self._patternSize, cv2.CALIB_CB_SYMMETRIC_GRID
        )

        for corner in corners.tolist():
            corners_box.append(corner[0])
        return corners_box

    def _get_four_top_corners(self, corners_box):
        """
        :param corners: self._total_corners corners ; type:list
        :return: 4个顶点
        """
        four_corners_index = [0, self.w - 1, -self.w, -1]

        left_up_corner = corners_box[four_corners_index[0]]
        right_up_corner = corners_box[four_corners_index[1]]
        left_down_corner = corners_box[four_corners_index[2]]
        right_down_corner = corners_box[four_corners_index[3]]

        return left_up_corner, right_up_corner, left_down_corner, right_down_corner

    def _midpoint_coordinates(self, p1, p2):
        x = (p1[0] + p2[0]) / 2
        y = (p1[1] + p2[1]) / 2
        middle_point = [x, y]
        return middle_point

    def _draw_contour(self, hierarchy):
        shape = hierarchy.shape
        new_hierarchy = hierarchy.reshape(
            int(shape[0] * shape[1] * shape[2] / 4), 4
        ).tolist()
        ji_he = set()
        for i in new_hierarchy:
            ji_he.add(i[3])

        index_list = []
        for value in ji_he:
            ls = []
            for i in new_hierarchy:
                if i[3] == value:
                    ls.append(i)
            if len(ls) == self._total_corners:
                index_list.append(ls)
                # print(f'内轮廓索引:{value}')
                return value

    def _inner_contour_short_edge_midpoint(self, approx):
        points_1 = approx.reshape(5, 2).tolist()
        points_2 = points_1.copy()

        for p1 in points_1:
            for p2 in points_2:
                if p1 != p2:
                    # self._draw_line(image,p1,p2)
                    distance = self._calc_dis2points(p1, p2)
                    if distance < 100:
                        midpoint = self._midpoint_coordinates(p1, p2)
                        return midpoint

    def _get_original_point(self, image):
        gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, thr1 = cv2.threshold(
            gray_image, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )

        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr1, cv2.MORPH_OPEN, kernel, iterations=5)
        contours, hierarchy = cv2.findContours(
            thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        index = self._draw_contour(hierarchy)

        # cv2.drawContours(image,contours,contourIdx=index,color=(255,0,0),thickness=3)

        cnt = contours[index]

        _epsilon = 0
        while True:
            approx = cv2.approxPolyDP(cnt, epsilon=_epsilon, closed=True)
            if len(approx) == 5:
                break
            else:
                _epsilon += 0.1

        _original_point = self._inner_contour_short_edge_midpoint(approx)

        return _original_point

    def _get_original_corner(self, original_point, corner1, corner2, corner3, corner4):
        """
        比较四个角点到original的距离,并返回最近的点
        :param original_point:
        :param p1: point1
        :param p2: point2
        :param p3: point3
        :param p4: point4
        :return: min distance corner
        """
        dict = {}

        d1 = self._calc_dis2points(original_point, corner1)
        d2 = self._calc_dis2points(original_point, corner2)
        d3 = self._calc_dis2points(original_point, corner3)
        d4 = self._calc_dis2points(original_point, corner4)

        dict[d1] = corner1
        dict[d2] = corner2
        dict[d3] = corner3
        dict[d4] = corner4

        min_distance = min(d1, d2, d3, d4)
        original_corner = dict[min_distance]

        return original_corner

    def _return_min_num_from_list(self, ls, n=3):
        ls_copy = copy.deepcopy(ls)
        min_number_ls = []
        min_index_ls = []
        for i in range(n):
            number = min(ls_copy)
            index = ls_copy.index(number)
            ls_copy[index] = max(ls_copy) + 1
            min_number_ls.append(number)
            min_index_ls.append(index)

        return min_number_ls, min_index_ls

    def _sort_dict(self, dict, flag="key"):
        """
        sorted dictionary
        :param dict: dictionary
        :param flag: 'key' or 'value'
        :return:
        """
        if flag == "key":
            sorted_dict = {}
            for key in sorted(dict):
                sorted_dict[key] = dict[key]
            return sorted_dict

        elif flag == "value":
            sorted_dict = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]))
            return sorted_dict

    def _build_coordinate_system(self, original_corner, corner_box):
        distance_list = []
        for corner in corner_box:
            distance = self._calc_dis2points(original_corner, corner)
            distance_list.append(distance)
        distance_list_copy = copy.deepcopy(distance_list)
        min_number, min_index = self._return_min_num_from_list(distance_list_copy, 3)

        p1 = corner_box[min_index[-2]]
        p2 = corner_box[min_index[-1]]

        vector1 = self._vector_A2B(p1, original_corner)
        vector2 = self._vector_A2B(p2, original_corner)

        result = np.cross(vector1, vector2)
        if result > 0:
            return p1, p2
        else:
            return p2, p1

    def _draw_calibration_board(self, image, corners_box):
        corners_box_array = (
            np.array(corners_box).reshape(self._total_corners, 1, 2).astype(np.float32)
        )
        cv2.drawChessboardCorners(
            image, self._patternSize, corners_box_array, corners_box_array is not None
        )

    def _sorted_corner_box(self, corners_box, original_corner, axis_point, flag):
        if flag == 0 or "x":
            distance_x_dict = {}
            for corner in corners_box:
                distance_x = self._distance_Point2Line(
                    original_corner, axis_point, target_point=corner
                )
                distance_x_dict[corners_box.index(corner)] = distance_x
            sorted_dict = self._sort_dict(distance_x_dict, flag="value")

            index_list_X = []
            for sorted_corner_index in sorted_dict:
                index_list_X.append(sorted_corner_index[0])

            new_corner_box_X = []
            for index in index_list_X:
                new_corner_box_X.append(corners_box[index])

            ls = np.array(new_corner_box_X).reshape(self.w, self.h, 2).tolist()

            new_corners_box = []
            for new_ls in ls:
                distance_list = {}
                for corner in new_ls:
                    distance = self._calc_dis2points(original_corner, corner)
                    distance_list[distance] = corner
                sorted_dict = self._sort_dict(distance_list, "key")
                for value in sorted_dict.values():
                    new_corners_box.append(value)
            return new_corners_box

        elif flag == 1 or "y":
            distance_y_dict = {}
            for corner in corners_box:
                distance_y = self._distance_Point2Line(
                    original_corner, axis_point, target_point=corner
                )
                distance_y_dict[corners_box.index(corner)] = distance_y
            sorted_dict = self._sort_dict(distance_y_dict, flag="value")
            index_list_Y = []
            for sorted_corner_index in sorted_dict:
                index_list_Y.append(sorted_corner_index[0])
            new_corner_box_Y = []
            for index in index_list_Y:
                new_corner_box_Y.append(corners_box[index])
            ls = np.array(new_corner_box_Y).reshape(self.w, self.h, 2).tolist()
            new_corners_box = []
            for new_ls in ls:
                distance_list = {}
                for corner in new_ls:
                    distance = self._calc_dis2points(original_corner, corner)
                    distance_list[distance] = corner
                sorted_dict = self._sort_dict(distance_list, "key")

                for value in sorted_dict.values():
                    new_corners_box.append(value)

            return new_corners_box

    def run(self, image, flag):
        """
        :param image:input image
        :param flag: 0:x_axis; 1:y_axis
        :return: new_corners_box and image
        """
        corners_box = self._get_corners_box(image)
        corner1, corner2, corner3, corner4 = self._get_four_top_corners(corners_box)
        original_point = self._get_original_point(image)
        original_corner = self._get_original_corner(
            original_point, corner1, corner2, corner3, corner4
        )
        axis_x_point, axis_y_point = self._build_coordinate_system(
            original_corner, corners_box
        )
        if flag == 0:
            new_corners_box = self._sorted_corner_box(
                corners_box, original_corner, axis_x_point, 0
            )
            print(f"sorted corners box:{new_corners_box}\n")
            self._draw_calibration_board(image, new_corners_box)
            return image, new_corners_box
        elif flag == 1:
            new_corners_box = self._sorted_corner_box(
                corners_box, original_corner, axis_y_point, 1
            )
            print(f"sorted corners box:{new_corners_box}\n")
            self._draw_calibration_board(image, new_corners_box)
            return image, new_corners_box


if __name__ == "__main__":
    start_time = time.time()
    ori_point = Sort_corners_box(w=7, h=7)

    for index in [0, 15, 19, 26, 43]:
        image_name = "{}_Y.png".format(index)
        image = cv2.imread(r"../left_img/{}.png".format(index))
        result_image, new_corners_box = ori_point.run(image, flag=0)
        cv2.imwrite("./test_result/{}".format(image_name), result_image)
        cv2.namedWindow("result_image", 0)
        cv2.imshow("result_image", result_image)
        cv2.waitKey(0)

    end_time = time.time()
    print("cost time: {}".format(end_time - start_time))
