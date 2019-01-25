
# Gaurav Kunte
# CS510 Introduction to Computer Vision
# Color Transfer

import cv2
import sys
import numpy as np
import math

def color_transfer_Lab(RGB_src, RGB_target):

	R = RGB_src[:,:,2]
	G = RGB_src[:,:,1]
	B = RGB_src[:,:,0]

	Rt = RGB_target[:,:,2]
	Gt = RGB_target[:,:,1]
	Bt = RGB_target[:,:,0]

	img_temp = np.array([[0.3811, 0.5783, 0.0402],
			     [0.1967, 0.7244, 0.0782],
			     [0.0241, 0.1288, 0.8444]])

	L = img_temp[0][0]*R + img_temp[0][1]*G + img_temp[0][2]*B
	M = img_temp[1][0]*R + img_temp[1][1]*G + img_temp[1][2]*B
	S = img_temp[2][0]*R + img_temp[2][1]*G + img_temp[2][2]*B
	


	Lt = img_temp[0][0]*Rt + img_temp[0][1]*Gt + img_temp[0][2]*Bt
	Mt = img_temp[1][0]*Rt + img_temp[1][1]*Gt + img_temp[1][2]*Bt
	St = img_temp[2][0]*Rt + img_temp[2][1]*Gt + img_temp[2][2]*Bt

	L = np.log10(L)
	M = np.log10(M)
	S = np.log10(S)

	Lt = np.log10(Lt)
	Mt = np.log10(Mt)
	St = np.log10(St)

	img_temp1 = np.array([[(1.0/math.sqrt(3)), 0.0, 0.0],
			     [0.0, (1.0/math.sqrt(6)), 0.0],
			     [0.0, 0.0, (1.0/math.sqrt(2))]])

	img_temp2 = np.array([[1.0, 1.0, 1.0],
			     [1.0, 1.0, -2.0],
			     [1.0, -1.0, 0.0]])

	img_temp3 = np.dot(img_temp1,img_temp2)

	l = img_temp3[0][0]*L + img_temp3[0][1]*M + img_temp3[0][2]*S
	a = img_temp3[1][0]*L + img_temp3[1][1]*M + img_temp3[1][2]*S
	b = img_temp3[2][0]*L + img_temp3[2][1]*M + img_temp3[2][2]*S

	lt = img_temp3[0][0]*Lt + img_temp3[0][1]*Mt + img_temp3[0][2]*St
	at = img_temp3[1][0]*Lt + img_temp3[1][1]*Mt + img_temp3[1][2]*St
	bt = img_temp3[2][0]*Lt + img_temp3[2][1]*Mt + img_temp3[2][2]*St

	lmean, amean, bmean = np.mean(l), np.mean(a), np.mean(b)

	l -= lmean
	a -= amean
	b -= bmean

	lstd, astd, bstd = np.std(l), np.std(a), np.std(b)
	ltstd, atstd, btstd = np.std(lt), np.std(at), np.std(bt)

	l = (ltstd/lstd) * l
	a = (atstd/astd) * a
	b = (btstd/bstd) * b

	ltmean, atmean, btmean = np.mean(lt), np.mean(at), np.mean(bt)

	l += ltmean
	a += atmean
	b += btmean

	matrix2 = np.array([[(math.sqrt(3)/3.0), 0.0, 0.0],
			     [0.0, (math.sqrt(6)/6.0), 0.0],
			     [0.0, 0.0, (math.sqrt(2)/2.0)]])

	matrix1 = np.array([[1.0, 1.0, 1.0],
			     [1.0, 1.0, -1.0],
			     [1.0, -2.0, 0.0]])

	LMStoRGB = np.array([[4.4679, -3.5873, 0.1193],
			     [-1.2186, 2.3809, -0.1624],
			     [0.0497, -0.2439, 1.2045]])

	matrix3 = np.dot(matrix1,matrix2)

	L_new =  matrix3[0][0]*l + matrix3[0][1]*a + matrix3[0][2]*b
	M_new =  matrix3[1][0]*l + matrix3[1][1]*a + matrix3[1][2]*b
	S_new =  matrix3[2][0]*l + matrix3[2][1]*a + matrix3[2][2]*b

	L_new = np.power(10.0, L_new)
	M_new = np.power(10.0, M_new)
	S_new = np.power(10.0, S_new)

	R_new = LMStoRGB[0][0]*L_new + LMStoRGB[0][1]*M_new + LMStoRGB[0][2]*S_new
	G_new = LMStoRGB[1][0]*L_new + LMStoRGB[1][1]*M_new + LMStoRGB[1][2]*S_new
	B_new = LMStoRGB[2][0]*L_new + LMStoRGB[2][1]*M_new + LMStoRGB[2][2]*S_new

	imgAfterLab = np.stack([B_new, G_new, R_new], axis = 2)
	
	return np.uint8(np.clip(imgAfterLab, 0, 255))

def color_transfer_Rgb(RGB_src, RGB_target):

	R = RGB_src[:,:,2]
	G = RGB_src[:,:,1]
	B = RGB_src[:,:,0]

	Rt = RGB_target[:,:,2]
	Gt = RGB_target[:,:,1]
	Bt = RGB_target[:,:,0]

	rmean, gmean, bmean = np.mean(R), np.mean(G), np.mean(B)

	R -= rmean
	G -= gmean
	B -= bmean

	rstd, gstd, bstd = np.std(R), np.std(G), np.std(B)
	rtstd, gtstd, btstd = np.std(Rt), np.std(Gt), np.std(Bt)

	R = (rtstd/rstd) * R
	G = (gtstd/gstd) * G
	B = (btstd/bstd) * B

	rtmean, gtmean, btmean = np.mean(Rt), np.mean(Gt), np.mean(Bt)

	R += rtmean
	G += gtmean
	B += btmean

	imgAfterRgb = np.stack([B, G, R], axis = 2)
	return np.uint8(np.clip(imgAfterRgb, 0, 255))



def color_transfer_CIECAM97s(RGB_src, RGB_target):

	R = RGB_src[:,:,2]
	G = RGB_src[:,:,1]
	B = RGB_src[:,:,0]

	Rt = RGB_target[:,:,2]
	Gt = RGB_https://start.ubuntu-mate.org/target[:,:,1]
	Bt = RGB_target[:,:,0]

	img_temp = np.array([[0.3811, 0.5783, 0.0402],
			     [0.1967, 0.7244, 0.0782],
			     [0.0241, 0.1288, 0.8444]])

	L = img_temp[0][0]*R + img_temp[0][1]*G + img_temp[0][2]*B
	M = img_temp[1][0]*R + img_temp[1][1]*G + img_temp[1][2]*B
	S = img_temp[2][0]*R + img_temp[2][1]*G + img_temp[2][2]*B
	


	Lt = img_temp[0][0]*Rt + img_temp[0][1]*Gt + img_temp[0][2]*Bt
	Mt = img_temp[1][0]*Rt + img_temp[1][1]*Gt + img_temp[1][2]*Bt
	St = img_temp[2][0]*Rt + img_temp[2][1]*Gt + img_temp[2][2]*Bt

	

	img_temp1 = np.array([[2.00, 1.00, 0.05],
			     [1.00, -1.09, 0.09],
			     [0.11, 0.11, -0.22]])


	A = img_temp1[0][0]*L + img_temp1[0][1]*M + img_temp1[0][2]*S
	c1 = img_temp1[1][0]*L + img_temp1[1][1]*M + img_temp1[1][2]*S
	c2 = img_temp1[2][0]*L + img_temp1[2][1]*M + img_temp1[2][2]*S

	At = img_temp1[0][0]*Lt + img_temp1[0][1]*Mt + img_temp1[0][2]*St
	c1t = img_temp1[1][0]*Lt + img_temp1[1][1]*Mt + img_temp1[1][2]*St
	c2t = img_temp1[2][0]*Lt + img_temp1[2][1]*Mt + img_temp1[2][2]*St

	amean, c1mean, c2mean = np.mean(A), np.mean(c1), np.mean(c2)

	A -= amean
	c1 -= c1mean
	c2 -= c2mean

	astd, c1std, c2std = np.std(A), np.std(c1), np.std(c2)
	atstd, c1tstd, c2tstd = np.std(At), np.std(c1t), np.std(c2t)

	A = (atstd/astd) * A
	c1 = (c1tstd/c1std) * c1
	c2 = (c2tstd/c2std) * c2

	atmean, c1tmean, c2tmean = np.mean(At), np.mean(c1t), np.mean(c2t)

	A += atmean
	c1 += c1tmean
	c2 += c2tmean

	img_temp_inv = np.linalg.inv(img_temp1)
	#print(img_temp_inv)

	LMStoRGB = np.array([[4.4679, -3.5873, 0.1193],
			     [-1.2186, 2.3809, -0.1624],
			     [0.0497, -0.2439, 1.2045]])

	L_new =  img_temp_inv[0][0]*A + img_temp_inv[0][1]*c1 + img_temp_inv[0][2]*c2
	M_new =  img_temp_inv[1][0]*A + img_temp_inv[1][1]*c1 + img_temp_inv[1][2]*c2
	S_new =  img_temp_inv[2][0]*A + img_temp_inv[2][1]*c1 + img_temp_inv[2][2]*c2

	R_new = LMStoRGB[0][0]*L_new + LMStoRGB[0][1]*M_new + LMStoRGB[0][2]*S_new
	G_new = LMStoRGB[1][0]*L_new + LMStoRGB[1][1]*M_new + LMStoRGB[1][2]*S_new
	B_new = LMStoRGB[2][0]*L_new + LMStoRGB[2][1]*M_new + LMStoRGB[2][2]*S_new

	imgAfterCIECAM97s = np.stack([B_new, G_new, R_new], axis = 2)
	
	return np.uint8(np.clip(imgAfterCIECAM97s, 0, 255))

	


if __name__ == "__main__":

	print('==================================================')
	print('PSU CS 410/510, Winter 2019, HW1: color transfer')
	print('==================================================')

	path_file_image_source = sys.argv[1]
	path_file_image_target = sys.argv[2]
	path_file_image_result_in_Lab = sys.argv[3]
	path_file_image_result_in_RGB = sys.argv[4]
	path_file_image_result_in_CIECAM97s = sys.argv[5]

	# ===== read input images
	# img_RGB_source: is the image you want to change the its color
	# img_RGB_target: is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)

	img_RGB_source = cv2.imread(filename = path_file_image_source).astype(np.float32)
	img_RGB_target = cv2.imread(filename = path_file_image_target).astype(np.float32)

	img_RGB_new_Lab = color_transfer_Lab(img_RGB_source, img_RGB_target)
	cv2.imwrite(path_file_image_result_in_Lab, (img_RGB_new_Lab))

	img_RGB_new_Rgb = color_transfer_Rgb(img_RGB_source, img_RGB_target)
	cv2.imwrite(path_file_image_result_in_RGB, (img_RGB_new_Rgb))

	img_RGB_new_CIECAM97s = color_transfer_CIECAM97s(img_RGB_source, img_RGB_target)
	cv2.imwrite(path_file_image_result_in_CIECAM97s, (img_RGB_new_CIECAM97s))



	

	
