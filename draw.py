import cv2
import numpy as np 

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
# mouse callback function


class cropIMG():
	def __init__(self, image):
		self.original = image
		self.img = self.original.copy()
		self.contours = []
		
	def interactive_drawing(self, event, x, y, flags, params):
		global ix, iy, drawing, mode

		if event == cv2.EVENT_LBUTTONDOWN:
			drawing = True
			ix, iy = x, y
			self.appanedPoint(x,y)
			cv2.line(self.img,(ix,iy),(x,y),(0,0,255),2)

		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
				if mode == True:
					cv2.line(self.img,(ix,iy),(x,y),(0,0,255),2) # draw line between former and present pixel
					ix, iy = x, y
					self.appanedPoint(x,y)
		
		elif event == cv2.EVENT_LBUTTONUP:
			drawing = False
			if mode == True:
				self.appanedPoint(x,y)
				cv2.line(self.img,(ix,iy),(x,y),(0,0,255),2)
	
	def cropBounfingrect(self):
		x,y,w,h = cv2.boundingRect(self.contours)
		self.masked_image = self.masked_image[y:y+h, x:x+w]

		
	def appanedPoint(self, px, py):
		px = px if px > 0 else 0
		py = py if py > 0 else 0
		self.contours.append([px, py])
		
	def getCorppedImg(self):
		return self.masked_image
	
	def crop(self):
		cv2.namedWindow('select')
		cv2.setMouseCallback('select', self.interactive_drawing)

		while(1):
			cv2.imshow('select', self.img)
			k = cv2.waitKey(1)&0xFF
			if k != 255:
				print k
			if k == 27: # Esc
				cv2.destroyAllWindows()
				break
			if k == 13: # Enter

				if self.contours == []:
					imsize = tuple(self.img.shape[1::-1])
					self.contours =  [[0,0], [imsize[0], 0], [imsize[0],imsize[1]], [0, imsize[1]]]

				self.contours = np.array(self.contours, dtype=np.int32) #pointsOf the polygon Like [[(10,10), (300,300), (10,300)]]
				mask = np.zeros(self.original.shape, dtype=np.uint8)
				white = (255, 255, 255)
				cv2.fillPoly(mask, np.array([self.contours], dtype=np.int32), white)

				# apply the mask
				self.masked_image = cv2.bitwise_and(self.original, mask)
				self.cropBounfingrect()
				cv2.imshow('masked image', self.masked_image)
				
			if k == ord("r") or k == ord("R"):
				print "reset"
				self.contours = []
			
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		

if __name__ == "__main__":
	myImg = cv2.imread(r'C:\Users\user\10_cropped.tif')
	myImg = cv2.resize(myImg, (0,0), fx=0.7, fy=0.7)
	cim = cropIMG(myImg)
	cim.crop()
	raw_input("Done")
