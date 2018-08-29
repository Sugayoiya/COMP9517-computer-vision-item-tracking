import numpy as np
import cv2,imutils as im
import sys

def getsize(img):
	h, w = img.shape[:2]
	return w,h

class image_feature_detector(object):
	SIFT = 0
	SURF = 1

	def __init__(self, feat_type, params = None):
		self.detector , self.norm = self.features_detector(feat_type=feat_type, params = params)

	def features_detector(self, feat_type = SIFT ,params = None):
		if feat_type == self.SIFT:
			if params is None:
				nfeatures = 0                 
				nOctaveLayers = 3                 
				contrastThreshold = 0.04                 
				edgeThreshold=10                 
				sigma=1.6             
			else:                 
				nfeatures = params["nfeatures"]                 
				nOctaveLayers = params["nOctaveLayers"]                 
				contrastThreshold = params["contrastThreshold"]                 
				edgeThreshold = params["edgeThreshold"]                 
				sigma = params["sigma"]
			
			detector = cv2. xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3, contrastThreshold=0.04,edgeThreshold=10, sigma=1.6)
			norm = cv2.NORM_L2
		elif feat_type == self.SURF:
			if params is None:                 
				hessianThreshold = 3000                 
				nOctaves = 1                 
				nOctaveLayers = 1                 
				upright = True                 
				extended = False             
			else:                 
				hessianThreshold = params["hessianThreshold"]                 
				nOctaves = params["nOctaves"]                 
				nOctaveLayers = params["nOctaveLayers"]                 
				upright = params["upright"] 
				extended = params["extended"]

			detector = cv2. xfeatures2d.SURF_create(hessianThreshold = hessianThreshold,nOctaves = nOctaves,nOctaveLayers = nOctaveLayers,upright = upright,extended = extended)
			norm = cv2.NORM_L2
		return detector,norm

if __name__ == "__main__":
	sift_detect = image_feature_detector(feat_type = 0)
	suft_detect = image_feature_detector(feat_type=1)

	image1 = cv2.imread('test.jpg')
	image2 = cv2.imread('test - Copy.jpg')
	image3 = im.resize(image1,width = 200)

	gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
	gray3 = cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)

	kp1, des1 = suft_detect.detector.detectAndCompute(gray1,None)
	kp2, des2 = suft_detect.detector.detectAndCompute(gray2,None)

	
	img1 = cv2.drawKeypoints(image1,kp1,None,(0,0,255),4)
	img2 = cv2.drawKeypoints(image2,kp2,None,(0,0,255),4)
	bf = cv2.BFMatcher()
	matches = bf.match(des1,des2)

	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,image3,flags=2)
	# sift
	cv2.imshow('a',img1)	
	cv2.imshow('b',img2)
	cv2.imshow('match',img3)
	cv2.imshow('out',image3)


	cv2.waitKey(0)
	cv2.destroyAllWindows()



			