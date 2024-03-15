import argparse
import sys

import numpy as np
import cv2 as cv
from util_func import *
import os
from os import listdir
import time

class SegmentImage:

    def __init__(self, image_path):
        super().__init__()
        self.img = cv.imread(image_path)
        self.time = time.time()
        
    def displayImage(self, seg_result):
        # Print the elapsed time in seconds
        print("--- %s seconds ---" % (time.time() - self.time))
        show_img("segmented image", seg_result)

    def getShapeWithHuMoment(self, img):
        maskCircle = np.loadtxt("circleNumpyArray.txt")

        # hough transform get mask
        mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (7, 7), 1.5)

        circlesB = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.3, 30, param1=150, param2=70, minRadius=0, maxRadius=0)

        if circlesB is not None:
            circlesB = np.uint16(np.around(circlesB))
            for c in circlesB[0, :]:
                cv.circle(mask, (c[0], c[1]), c[2], (255, 0, 255), -1)

        d2 = cv.matchShapes(maskCircle, mask, cv.CONTOURS_MATCH_I2, 0)

        if(d2 < 0.1):
            return "circle"

        return False

    def houghSegmentationSingle (self, contour_image):
        """
        contour_image = images having contour
        ori_image = original image
        """
        img = contour_image.copy()
        ori_image = self.img.copy()
        
        mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8) #create mask

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (7, 7), 1.5)

        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.3, 30, param1=150, param2=70, minRadius=0, maxRadius=0)

        # circle is empty
        if circles is None:
            return contour_image

        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv.circle(mask, (c[0], c[1]), c[2], (255, 0, 255), -1)


        mask = self.dilation(mask)
        masked = cv.bitwise_and(ori_image, ori_image, mask=mask)

        return masked

    def dilation(self,img):
        kernel = np.ones((5,5),np.uint8)

        img_copy = img.copy()
        dilation = cv.dilate(img,kernel,iterations = 1)
        return dilation

    def segmentTrafficSign(self):
        

        
        if(self.img is None):
            print("no image found")
            return
        img = self.img.copy()

        bilateral_filtered_image = cv.bilateralFilter(img.copy(), 5, 175, 175)

        # Perform edge detection on the segmented traffic sign
        edges = cv.Canny(bilateral_filtered_image, 75, 200)

        # Find contours in the edge-detected segmented traffic sign
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        sign_mask = np.zeros_like(img)

        if contours:
            # oval only
            contour_list = []
            for contour in contours:
                approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True),True)
                area = cv.contourArea(contour)
                if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
                    contour_list.append(contour)            
            if contour_list:
                # Filter contours based on area to find the largest contour (traffic sign)
                largest_contour = max(contour_list, key=cv.contourArea)
                cv.drawContours(sign_mask, [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED)

                #successfully segment oval
                if self.getShapeWithHuMoment(sign_mask):
                    seg_traffic_sign = cv.bitwise_and(img, sign_mask)
                    seg_result = seg_traffic_sign
                    self.displayImage(seg_result)
                    return

            # for other shape 
            # Filter contours to find the largest contour in the edge-detected segmented traffic sign
            largest_contour_seg = max(contours, key=cv.contourArea)

            # Draw the largest contour on a black image
            sign_mask = np.zeros_like(img)
            cv.drawContours(sign_mask, [largest_contour_seg], -1, (255, 255, 255), thickness=cv.FILLED)          

        # get segmented image here    
        seg_traffic_sign = cv.bitwise_and(img, sign_mask)

        # color threshold segmentation
        img = self.img.copy()

        # Convert image from BGR to the HSV color space
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Define upper and lower bounds for the light blue color in HSV
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([110, 255, 255])

        # Define upper and lower bounds for the red color in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Define upper and lower bounds for the yellow color in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create binary masks for each color
        mask_blue = cv.inRange(hsv_image, lower_blue, upper_blue)
        mask_red1 = cv.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv.inRange(hsv_image, lower_red2, upper_red2)
        mask_yellow = cv.inRange(hsv_image, lower_yellow, upper_yellow)

        # Combine the masks for different colors
        combined_mask = mask_blue | mask_red1 | mask_red2 | mask_yellow

        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)

        # Find contours in the binary mask
        contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            
            # Filter contours based on area to find the largest contour (traffic sign)
            largest_contour = max(contours, key=cv.contourArea)

            # Create a mask for the traffic sign region
            sign_mask = np.zeros_like(img)
            cv.drawContours(sign_mask, [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED)

            # Extract the segmented traffic sign from the original image using the mask
            seg_traffic_sign2 = cv.bitwise_and(img, sign_mask)

            #merge edge detection and color threshold
            result = cv.add(seg_traffic_sign, seg_traffic_sign2)

            #check if merge brings out result
            #check if it is circle using hough transform
            houghImg = self.houghSegmentationSingle(result)

            if self.getShapeWithHuMoment(houghImg):
                seg_result = houghImg
                self.displayImage(seg_result)
                return
            else:
                #continue do other shape
                #return edge detection result
                seg_result = seg_traffic_sign
                
        else:
            # return edge detection result
            seg_result = seg_traffic_sign

            #display
        self.displayImage(seg_result)
        

#main function        
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) != 3:
        print("invalid image filepath.")
    else:
        image_path = sys.argv[2]
        image_path = image_path.replace("\/", "/")
        image_path = image_path.replace("\"","")
        segmentImage = SegmentImage(image_path)
        segmentImage.segmentTrafficSign()
        
