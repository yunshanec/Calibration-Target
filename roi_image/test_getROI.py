from toolbox import imgproctool
import cv2

for i in range(1,25):
    img = cv2.imread('../original_negative/{}.png'.format(i))
    roi = [776, 629, 1481, 1015]

    cv2.namedWindow("src", 0)
    cv2.imshow("src", img)
    cv2.waitKey(0)

    _, roi_img = imgproctool.getRoiImg(img, roi=roi, roiType=imgproctool.ROI_TYPE_XYXY)
    cv2.namedWindow("Roi_img",0)
    cv2.imshow("Roi_img",roi_img)
    cv2.waitKey(0)

    cv2.imwrite('./roi_image/negative_ROI/{}.png'.format(i),roi_img)