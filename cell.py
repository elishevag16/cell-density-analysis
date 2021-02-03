# C:\Python27\python.exe "$(FULL_CURRENT_PATH)"

import numpy as np
import cv2
import os, shutil
import ctypes
from draw import *
import re
import ConfigParser
import HSVsample

def fast():
	img = example.copy()
	height, width, depth = img.shape
	img[0:height, 0:width//4, 0:depth] = 0
	return img 
	
def slow(img):
	height, width, depth = img.shape
	for i in range(0, height):            
		for j in range(0, width): 
			for k in range(0, depth): 
				img[i, j, k] = 0
	return img

def ROI_select(image, points):

	mask = np.zeros(image.shape, dtype=np.uint8)
	roi_corners = np.array(points, dtype=np.int32)
	white = (255, 255, 255)
	cv2.fillPoly(mask, roi_corners, white)

	# apply the mask
	masked_image = cv2.bitwise_and(image, mask)

	cv2.imshow('masked image', masked_image)
	cv2.waitKey()
	cv2.destroyAllWindows()


def gray_treshold(img, t):
	mask = cv2.inRange(img, 0, t)
	output = cv2.bitwise_and(img, img, mask = mask)

	return output
	
def RGBboundaries(img, boundaries):
	(lower, upper) = boundaries[0]
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	mask = cv2.inRange(img, lower, upper)
	output = cv2.bitwise_and(img, img, mask = mask)
	return output

def HSVboundaries(img, boundaries):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	
	lower = boundaries[0]
	upper = boundaries[1]
	
	print "lower = ", lower
	print "upper = ", upper
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower, upper)
	# Bitwise-AND mask and original image
	output = cv2.bitwise_and(img, img, mask= mask)
	return output, mask	
	
def getBoundaries():
	HSVonf = ConfigParser.ConfigParser()
	HSVonf.readfp(open('HSV.ini'))
	selected =  HSVonf.get('HSV', 'selected')
	lower = np.array([int(x) for x in HSVonf.get(selected, 'HSV_min').split()])
	upper = np.array([int(x) for x in HSVonf.get(selected, 'HSV_max').split()])
	return [lower, upper]


def saveBoundaries(outdir):
	shutil.copyfile('HSV.ini', os.path.join(outdir, 'HSV.ini'))
	
def samplePurple(img):
	max = 1
	min = 0

	r = [255, 0]
	g = [255, 0]
	b = [255, 0]
	for i in range(rows):
		for j in range(cols):
			k = imCrop[i,j]
			print k
			red = k[2]
			green = k[1]
			blue = k[0]
			if red > 200 and green > 200 and blue > 200:
				continue
			if r[min] > red: r[min] = red
			if r[max] < red: r[max] = red
			if g[min] > green: g[min] = green
			if g[max] < green: g[max] = green
			if b[min] > blue: b[min] = blue
			if b[max] < blue: b[max] = blue

	print "red", r
	print "green", g
	print "blue", b		

	p_boundaries = [([b[min], g[min], r[min]], [b[max], g[max], r[max]])]
	return p_boundaries

def	walkOnDir():
	script_dir = ""
	try:	 
		script_dir = (os.path.dirname(os.path.realpath(__file__)))
	except:
		script_dir = (os.path.dirname(os.path.realpath('__file__')))
	files_list = os.listdir(script_dir)

	for file in reversed(files_list):
		if not(file.endswith(".tif")):
			files_list.remove(file)

	return script_dir
	
def removeSpaces(file):
	newName = file.replace(" ", "_")
	print file, newName
	os.rename(file, newName)
	return newName

def cropRect(img):
	# Select ROI
	r = cv2.selectROI(img)
	 
	# Crop image
	imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
	return imCrop

def calcFilteredErea(org, filtered):
	
	if len(org.shape) == 3:
		org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)

	if len(filtered.shape) == 3:
		filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
		
	nzCount_org = cv2.countNonZero(org)
	nzCount_filtered = cv2.countNonZero(filtered)
	
	percent = (float(nzCount_filtered)/float(nzCount_org)) * 100.0
	print "%s%% were selected" % percent
	return percent

def cv_size(img):
	return tuple(img.shape[1::-1])
	
def resizeToWin(img):
	HIGHT = 0
	WIDTH = 1
	user32 = ctypes.windll.user32
	screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
	im_size = cv_size(img)

	hight_ratio = float(screensize[HIGHT])/float(im_size[HIGHT])
	width_ratio = float(screensize[WIDTH])/float(im_size[WIDTH])

	factor = width_ratio if width_ratio < hight_ratio else hight_ratio
	return cv2.resize(img, (0,0), fx=factor, fy=factor)
	
def filterTest(original_img, mask):
	inv_mask = cv2.bitwise_not(mask)
	output = cv2.bitwise_and(original_img, original_img, mask = inv_mask)

	return output

def getFileInfo(fileName):
	pattern = "(?P<Age>[O,Y]{1})(?P<day>\d+)(?P<sample>[C,T]\d+)(?P<slide>S\d+[a-z]{1})_(?P<area>.*)(?P<zoom>X\d{2})"
	m = re.search(pattern, fileName)
	if m == "NONE":
		return False
	try:
		fileInfo = "%s,%s,%s,%s,%s,%s" % (m.group("Age"), m.group("day"), m.group("sample"), m.group("slide"), m.group("area") ,m.group("zoom"))
	except:
		fileInfo = "%s,%s,%s,%s,%s,%s" % ("fail", "fail", "fail", "fail", "fail", "fail")
	return fileInfo
	

def findCropedFiles(dir):
	croped = []
	for root, dirs, files in os.walk(dir):
		for name in files:
			if "_cropped" in name:
				croped.append(os.path.join(root,name))
	return croped
	
def filterCropedImg(file, outdir = None, silent = False):
	p_boundaries = HSVsample.selectHSV(file)

	imPath = file
	img = cv2.imread(imPath)
	imPath_base = os.path.basename(imPath)
	if silent == False:
		img = resizeToWin(img)
		cim = cropIMG(img)
		cim.crop()
		img = cim.getCorppedImg()
		
	
	p_filtered, mask = HSVboundaries(img, p_boundaries)
	percentOfCell = calcFilteredErea(img, p_filtered)
	invert = filterTest(img, mask)
	if silent == False:
		cv2.imshow("results", np.hstack([img, invert]))
		cv2.imshow("results", np.hstack([img, p_filtered]))
		cv2.imshow("results", np.hstack([p_filtered, invert]))
		cv2.waitKey(0)
	
	newfile_analyzed = imPath_base.split(".")[0] + "_analyzed." + imPath_base.split(".")[1]
	newfile_cropped = imPath_base.split(".")[0] + "_cropped." + imPath_base.split(".")[1]
	newfile_inv = imPath_base.split(".")[0] + "_invert." + imPath_base.split(".")[1]
	
	if outdir == None:
		outdir = os.path.dirname(imPath)
		
	savePath = os.path.join(outdir, newfile_analyzed)
	print "save ", savePath
	cv2.imwrite(savePath, np.hstack([p_filtered, img]))
	
	savePath = os.path.join(outdir, newfile_cropped)
	print "save ", savePath
	cv2.imwrite(savePath, img)

	savePath = os.path.join(outdir, newfile_inv)
	print "save ", savePath
	cv2.imwrite(savePath, np.hstack([p_filtered, invert]))
	
	cvsName = os.path.join(outdir,"results.csv")
	if not(os.path.isfile(cvsName)):
		with open(os.path.join(walkOnDir(), cvsName), "a+") as f:
			f.write("img path,savePath,HSV,age,day,sample,slide,area,zoom,percentOfCell\n")

	with open(os.path.join(walkOnDir(), cvsName), "a+") as f:
		hsv = "[ "+ " ".join(str(x) for x in p_boundaries[0]) +" ]"
		hsv += "[ " + " ".join(str(x) for x in p_boundaries[1])  +" ]"
		toCSV = "%s,%s,%s,%s,%s\n" % (imPath, savePath, hsv,getFileInfo(imPath_base),percentOfCell)
		f.write(toCSV)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	input = raw_input("Please enter a path to file or directory:")
	if os.path.isdir(input):
		outdir = raw_input("Please enter a path to outdir: ")

		list = findCropedFiles(input)
		saveBoundaries(outdir)
		for file in list:
			print "processing ", file
			filterCropedImg(file, outdir, True)
			
	if os.path.isfile(input):
		filterCropedImg(input)	

	raw_input("Done")
