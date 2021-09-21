import cv2
import numpy as np
from PIL import Image

class EffectMaker:
    
    def __init__(self,name): # Name of the file without .png
        # Photo must be .png file
        self.name = str(name)
        self.filename = self.name + '.png'
    
    def _edge_mask(self,img, line_size, blur_value):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
        return edges

    def _color_quantization(self,img, k):
        # Transform the image
        data = np.float32(img).reshape((-1, 3))

        # Determine criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

        # Implementing K-Means
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result

    def pixelize(self,type_res):
        myImage = Image.open(self.name + '.png')
        smolImage = myImage.resize((type_res,type_res),Image.BILINEAR)
        result = smolImage.resize(myImage.size,Image.NEAREST)
        result.save(self.name +'_pixelized' + '.png')

    def pencil_scetch(self):
        img = cv2.imread(self.filename)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        inverted_grey_img = 255 - gray_img
        blurred_img = cv2.GaussianBlur(inverted_grey_img,(21,21),0)
        inverted_blurred_img = 255 - blurred_img
        pencil_scetch_img = cv2.divide(gray_img,inverted_blurred_img,scale = 256.0)
        cv2.imwrite(self.name +'_scetched' + '.png',pencil_scetch_img)

    def negative(self):
        img = cv2.imread(self.filename)
        filtered_image = cv2.bitwise_not(img)
        cv2.imwrite(self.name +'_negative' + '.png',filtered_image)

    def black_white(self):
        img = cv2.imread(self.filename)
        filtered_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.cvtColor(filtered_image,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(self.name +'_blackwhite' + '.png',filtered_image)

    def sepia(self):
        img = cv2.imread(self.filename)
        kernel = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])

        filtered_image = cv2.filter2D(img,-1,kernel)
        cv2.imwrite(self.name +'_sepia' + '.png',filtered_image)

    def emboss(self):
        img = cv2.imread(self.filename)
        kernel = np.array([[0, -1, -1],
                        [1, 0, -1],
                        [1, 1, 0]])
        filtered_image = cv2.filter2D(img,-1,kernel)
        cv2.imwrite(self.name +'_emboss' + '.png',filtered_image)

    def gaussian_blur(self):
        img = cv2.imread(self.filename)
        filtered_image = cv2.GaussianBlur(img,(41,41),0)
        cv2.imwrite(self.name +'_gBlur' + '.png',filtered_image)

    def median_blur(self):
        img = cv2.imread(self.filename)
        filtered_image = cv2.medianBlur(img,41)
        cv2.imwrite(self.name +'_mBlur' + '.png',filtered_image)

    def oilpaint(self):
        img = cv2.imread(self.filename)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
        morph = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
        result = cv2.normalize(morph,None,20,255, cv2.NORM_MINMAX)
        cv2.imwrite(self.name +'_oilpaint' + '.png',result)

    def convert_into_cartoon(self):
        img = cv2.imread(self.filename)
        line_size = 7
        blur_value = 7
        edges = self._edge_mask(img, line_size, blur_value)
        total_color = 9
        img = self._color_quantization(img, total_color)
        blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)
        cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
        cv2.imwrite(self.name +'_cartoon' + '.png',cartoon)

    def color_paletee(self):
        img = cv2.imread(self.filename)
        line_size = 7
        blur_value = 7
        total_color = 9
        edges = self._edge_mask(img, line_size, blur_value)
        img = self._color_quantization(img, total_color)
        cv2.imwrite(self.name +'_paleete' + '.png',img)

    def bilateral_filter(self):
        img = cv2.imread(self.filename)
        line_size = 7
        blur_value = 7
        total_color = 9
        edges = self._edge_mask(img, line_size, blur_value)
        img = self._color_quantization(img, total_color)
        blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)
        cv2.imwrite(self.name +'_bilateral' + '.png',blurred)

