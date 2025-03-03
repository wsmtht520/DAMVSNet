import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse, os, sys, time, gc, datetime
from torchvision import transforms
from PIL import Image 


"""
    以下都是测试代码
"""


# 示例1：使用sobel算子对图像进行边缘检测，但是是对彩色三通道图像，而不是灰度图像
# Read the color image
# image_path = "./outputs_test0103_depthpriorN5_adaptCost/scan77/images/00000001.jpg"
# original_image = cv2.imread(image_path)

# # Apply the Sobel operator to each color channel
# sobel_x_r = cv2.Sobel(original_image[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
# sobel_y_r = cv2.Sobel(original_image[:, :, 0], cv2.CV_64F, 0, 1, ksize=3)

# sobel_x_g = cv2.Sobel(original_image[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
# sobel_y_g = cv2.Sobel(original_image[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)

# sobel_x_b = cv2.Sobel(original_image[:, :, 2], cv2.CV_64F, 1, 0, ksize=3)
# sobel_y_b = cv2.Sobel(original_image[:, :, 2], cv2.CV_64F, 0, 1, ksize=3)

# # Compute the magnitude of the gradient for each channel
# magnitude_r = np.sqrt(sobel_x_r**2 + sobel_y_r**2)
# magnitude_g = np.sqrt(sobel_x_g**2 + sobel_y_g**2)
# magnitude_b = np.sqrt(sobel_x_b**2 + sobel_y_b**2)

# # Combine the magnitudes to get a single-channel result
# magnitude_combined = np.sqrt(magnitude_r**2 + magnitude_g**2 + magnitude_b**2)

# # Normalize the magnitude to 8-bit for display
# magnitude_combined = cv2.normalize(magnitude_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# # Display the original image and the combined gradient magnitude
# plt.figure(figsize=(10, 5))

# plt.subplot(121)
# plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Color Image')

# plt.subplot(122)
# plt.imshow(magnitude_combined, cmap='gray')
# # plt.imshow(magnitude_combined)
# plt.title('Combined Sobel Gradient Magnitude')
# plt.savefig("./combined_sobel_scan77_2.jpg")

# plt.show()



# 示例2：使用sobel算子对图像进行边缘检测，对灰度图像
# Read the image in color
# image_path = "./outputs_test0103_depthpriorN5_adaptCost/scan33/images/00000001.jpg"
# original_image = cv2.imread(image_path)

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# # Apply the Sobel operator
# sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

# # Compute the magnitude of the gradient
# # magnitude = cv2.magnitude(sobel_x, sobel_y)
# magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  

# Display the original image, grayscale image, and gradient magnitude
# plt.figure(figsize=(12, 4))

# plt.subplot(131)
# plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')

# # plt.subplot(132)
# # plt.imshow(gray_image, cmap='gray')
# # plt.title('Grayscale Image')

# plt.subplot(133)
# plt.imshow(magnitude, cmap='gray')
# plt.title('Sobel Gradient Magnitude')
# plt.savefig("./combined_sobel_scan33_3.jpg")

# plt.show()


# 示例3：将sobel处理后的结果与原图像进行堆叠
# Read the image
# image_path = "./outputs_test0103_depthpriorN5_adaptCost/scan13/images/00000001.jpg"
# original_image = cv2.imread(image_path)

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# # Apply the Sobel operator to compute gradients
# sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

# # Compute the magnitude of the gradient
# sobel_grad = np.sqrt(sobel_x**2 + sobel_y**2)

# # Normalize the magnitude to 8-bit for display
# magnitude = cv2.normalize(sobel_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# # Create a 3-channel image with edges in red
# edges_colored = cv2.merge([magnitude, np.zeros_like(magnitude), np.zeros_like(magnitude)])

# # Overlay edges on the original image
# result_image = cv2.addWeighted(original_image, 1, edges_colored, 0.7, 0)

# # Display the original image, edges, and result
# plt.figure(figsize=(12, 4))

# plt.subplot(131)
# plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')

# plt.subplot(132)
# plt.imshow(sobel_grad, cmap="gray")
# plt.title('Sobel Edges')

# plt.subplot(133)
# plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
# plt.title('fused original image and sobel iamge')
# plt.savefig("./combined_sobel_scan13_4.jpg")

# plt.show()


# 示例4：对于PIL读取到的图像，基于Sobel算子实现边缘检测
# Load the image using PIL
# image_path = "./outputs_test0103_depthpriorN5_adaptCost/scan11/images/00000001.jpg"
# original_image = Image.open(image_path)

# # Convert the image to grayscale
# gray_image = original_image.convert('L')

# # Apply the Sobel filter for edge detection
# sobel_image = gray_image.filter(ImageFilter.FIND_EDGES)

# # Convert PIL image to NumPy array for visualization
# sobel_array = np.array(sobel_image)

# # Display the original image and Sobel edge-detected image
# plt.figure(figsize=(10, 5))

# plt.subplot(121)
# plt.imshow(np.array(original_image), cmap='gray')
# plt.title('Original Image')

# plt.subplot(122)
# plt.imshow(sobel_array, cmap='gray')
# plt.title('Sobel Edge Detection')
# plt.savefig("./combined_sobel_scan11_5.jpg")

# plt.show()


# 示例5：对于PIL读取到的图像，先转换为numpy格式，然后再使用opencv的sobel算子进行边缘检测
# Load the image using PIL
# image_path = "../pic/scan77/00000001.jpg"
image_path = "../pic/building3.png"
pil_image = Image.open(image_path)

# Convert PIL image to NumPy array
# normail 0~255 to 0~1
numpy_image = np.array(pil_image, dtype=np.float32)/255.
# numpy_image = np.array(pil_image, dtype=np.float32)

# Convert the image to grayscale if it's a color image
if len(numpy_image.shape) == 3:
    gray_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
else:
    gray_image = numpy_image

# Apply Sobel operator for edge detection
sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

# Compute the magnitude of the gradient
magnitude = np.sqrt(sobel_x**2 + sobel_y**2) * 255.

# # 使用opencv保存图像进行查看比plt要方便
# cv2.imwrite("./sobel_scan77_1.jpg", magnitude)
cv2.imwrite("./sobel_building_1.jpg", magnitude)
plt.imsave("./sobel_building_2.jpg", magnitude, cmap='binary')
# print(magnitude.shape)  # (864, 1152)
# # grad_img = magnitude*255.
# # cv2.imwrite("./combined_sobel_scan9_88.jpg", grad_img)
# # print(grad_img)
# # print(magnitude)
# # print(magnitude.shape)  # (864, 1152) ndarray
# img_contain = []
# # AttributeError: 'NoneType' object has no attribute 'append'
# # img_contain = img_contain.append(magnitude)
# img_contain.append(magnitude)
# img_contain.append(magnitude)
# img_contain.append(magnitude)
# imgs = np.stack(img_contain)[:,:,:,None].transpose([0,3,1,2])
# print(imgs.shape)  # (3, 1, 864, 1152)
# print(type(imgs))  # <class 'numpy.ndarray'>
# imgs = np.stack(img_contain).transpose([0,1,2])
# print(imgs.shape)      # (3, 864, 1152)
# print(type(imgs))     # <class 'numpy.ndarray'>
# result = imgs[:,:,:,None]  
# print(result.shape)     # (3, 864, 1152, 1)
# results = result.transpose([3,0,1,2])
# print(results.shape)    # (1, 3, 864, 1152)
# # AttributeError: 'numpy.ndarray' object has no attribute 'unsqueez
# # grad_img = imgs.unsqueeze(0)




# # Normalize the magnitude to 8-bit for display
# magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# # Display the original image, grayscale image, and Sobel edge-detected image
# plt.figure(figsize=(12, 4))

# plt.subplot(131)
# plt.imshow(numpy_image)
# plt.title('Original Image')

# plt.subplot(132)
# plt.imshow(gray_image, cmap='gray')
# plt.title('Grayscale Image')

# plt.subplot(133)
# plt.imshow(magnitude, cmap='gray')
# plt.title('Sobel Edge Detection')
# # plt.savefig("./combined_sobel_scan77_6.jpg")

# plt.show()

