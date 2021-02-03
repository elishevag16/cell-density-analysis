import cv2
import numpy as np
import ConfigParser



def toggelSW(x):
	return x
	
def nothing(x):
    pass
	
def selectHSV(imPath):
	HSVonf = ConfigParser.ConfigParser()
	HSVonf.readfp(open('HSV.ini'))
	lower = ([int(x) for x in HSVonf.get("last", 'HSV_min').split()])
	upper = ([int(x) for x in HSVonf.get("last", 'HSV_max').split()])

	cv2.namedWindow('result')
	h = 0
	s = 1
	v = 2

	cv2.createTrackbar('h_min', 'result', 0, 179, nothing)
	cv2.createTrackbar('s_min', 'result', 0, 255, nothing)
	cv2.createTrackbar('v_min', 'result', 0, 255, nothing)

	cv2.createTrackbar('h_max', 'result', 0, 179, nothing)
	cv2.createTrackbar('s_max', 'result', 0, 255, nothing)
	cv2.createTrackbar('v_max', 'result', 0, 255, nothing)


	cv2.setTrackbarPos('h_min', 'result', lower[h])
	cv2.setTrackbarPos('s_min', 'result', lower[s])
	cv2.setTrackbarPos('v_min', 'result', lower[v])
	cv2.setTrackbarPos('h_max', 'result', upper[h])
	cv2.setTrackbarPos('s_max', 'result', upper[s])
	cv2.setTrackbarPos('v_max', 'result', upper[v])

	# create switch for ON/OFF functionality
	switch = '0 : INV_OFF \n1 : INV_ON'
	cv2.createTrackbar(switch, 'result', 0, 1, nothing)

	img = cv2.imread(imPath)
	cv2.namedWindow('result',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('result',300,300)
	# img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	frame = img

	while(1):

		# _, frame = cap.read()


		#converting to HSV
		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

		# get info from track bar and appy to result
		h = cv2.getTrackbarPos('h_min','result')
		s = cv2.getTrackbarPos('s_min','result')
		v = cv2.getTrackbarPos('v_min','result')
		
		h_max = cv2.getTrackbarPos('h_max', 'result')
		s_max = cv2.getTrackbarPos('s_max', 'result')
		v_max = cv2.getTrackbarPos('v_max', 'result')
		
		
		# Normal masking algorithm
		lower_blue = np.array([h,s,v])
		upper_blue = np.array([h_max, s_max, v_max])
		mask = cv2.inRange(hsv,lower_blue, upper_blue)
		if(cv2.getTrackbarPos(switch,'result') == 0):
			result = cv2.bitwise_and(frame,frame,mask = mask)
		elif(cv2.getTrackbarPos(switch,'result') == 1):
			inv_mask = cv2.bitwise_not(mask)
			result = cv2.bitwise_and(frame,frame,mask = inv_mask)


		cv2.imshow('im',result)

		k = cv2.waitKey(5) & 0xFF
		if k == 27: # Esc
			break
		if k == ord("r"):
			print lower_blue, upper_blue
		
	# cap.release()

	cv2.destroyAllWindows()
	
	config = ConfigParser.RawConfigParser()
	config.add_section('last')
	config.set('last', 'HSV_min', " ".join(str(x) for x in lower_blue.tolist()))
	config.set('last', 'HSV_max', " ".join(str(x) for x in upper_blue.tolist()))
	with open('HSV.ini', 'wb') as configfile:
		config.write(configfile)
	
	return [lower_blue, upper_blue]


if __name__ == "__main__":
	imPath = r'C:\Users\user\Desktop\10.tif'
	lower, upper = selectHSV(imPath)
	print lower, upper
	raw_input("Done")