
import numpy as np
import argparse, os, sys, re,time, gc, datetime,cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


from utilsme.io_utils import save_pfm, read_pfm, load_pfm



# parser = argparse.ArgumentParser(description="UniMVSNet args")

# # visualization
# parser.add_argument("--vis", action="store_true")
# parser.add_argument('--depth_path', type=str, default="./outputs_test1113_transformer/scan33/depth_est/00000000.pfm")
# parser.add_argument('--depth_img_save_dir', type=str, default="./outputs_test1113_transformer/scan33/depth_est_color") 

# args = parser.parse_args()

# code from Uni-MVSNet
# 这个从pfm格式读取出来的深度，将其可视化出来的伪深度图不是太好看
def visualization(self):

    import matplotlib as mpl
    import matplotlib.cm as cm
    from PIL import Image

    save_dir = self.args.depth_img_save_dir
    depth_path = self.args.depth_path

    depth, scale = read_pfm(depth_path)
    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)
    im.save(os.path.join(save_dir, "depth.png"))

    print("Successfully visualize!")


# 自己写的接口：读取深度图然后转换成伪深度
def convertPNG(pngfile,outdir):
    #读取16位深度图（像素范围0～65535），并将其转化为8位（像素范围0～255）
    uint16_img = cv2.imread(pngfile, -1)    #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取，否则默认的读取类型为8位。
    uint16_img -= uint16_img.min()
    uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
    uint16_img *= 255
    #使得越近的地方深度值越大，越远的地方深度值越小，以达到伪彩色图近蓝远红的目的。
    uint16_img = 255 - uint16_img
 
    # cv2 中的色度图有十几种，其中最常用的是 cv2.COLORMAP_JET，蓝色表示较高的深度值，红色表示较低的深度值。
    # cv.convertScaleAbs() 函数中的 alpha 的大小与深度图中的有效距离有关，如果像我一样默认深度图中的所有深度值都在有效距离内，并已经手动将16位深度转化为了8位深度，则 alpha 可以设为1。
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(uint16_img,alpha=1),cv2.COLORMAP_JET)
    #convert to mat png
    im=Image.fromarray(im_color)
    #save image
    im.save(os.path.join(outdir,os.path.basename(pngfile)))

def DepthMapPseudoColorize(depth_map,output_path, max_depth = None, min_depth = None):
    # Load 16 units depth_map(element ranges from 0~65535),
    # Convert it to 8 units (element ranges frome 0~255) and pseudo colorize
    if isinstance(depth_map, np.ndarray):
        uint16_img =  depth_map
    # Indicate the argument load mode -1, otherwise loading default 8 units
    if isinstance(depth_map, str):
        uint16_img = cv2.imread(depth_map, -1)
    if None == max_depth:
        max_depth = uint16_img.max()
    if None == min_depth:
        min_depth = uint16_img.min()

    uint16_img -= min_depth
    uint16_img = uint16_img / (max_depth - min_depth)
    uint16_img *= 255

    # cv2.COLORMAP_JET, blue represents a higher depth value, and red represents a lower depth value
    # The value of alpha in the cv.convertScaleAbs() function is related to the effective distance in the depth map. If like me, all the depth values
    # in the default depth map are within the effective distance, and the 16-bit depth has been manually converted to 8-bit depth. , then alpha can be set to 1.
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(uint16_img,alpha=1),cv2.COLORMAP_JET)
    #convert to mat png
    im=Image.fromarray(im_color)
    im.save(output_path)


#@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
    
# read_pfm(): data, scale
# load_pfm(): data
# if __name__ == '__main__':
    
#     # depth = load_pfm(args.depth_path)
#     depth, scale = read_pfm(args.depth_path)  # 两者均可以
#     mi_d = np.min(depth[depth>0])
#     ma_d = np.max(depth)
#     depth = (depth-mi_d)/(ma_d-mi_d+1e-8) # 归一化
#     depth = (255*depth).astype(np.uint8) # 映射到0~255
#     # 没有必要使用cv2.convertScaleAbs，因为MVSNet的重建深度都是自己想要的有效深度范围内，故没有必要再进行缩放取绝对值再取uint8
#     # im_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=1), cv2.COLORMAP_JET) # 转换为伪彩色图
#     im_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
#     outdir = "./outputs_test1113_transformer/scan33"
#     cv2.imwrite(os.path.join(outdir, "depth_est/0000002.jpg"), im_color)

#     # im = Iamge.fromarry(im_color)  # 不建议opencv和PIL库混合使用
#     # im.save(os.path.join(outdir, "depth_est"))


if __name__ == '__main__':
    
    # input: 手动输入路径
    # 读取output/scan13/depth_est目录下的pfm文件,将所有的pfm格式的深度进行读取,并转换为伪深度jpg图像格式（文件名不变）。
    # 同时在depth_est同一级目录创建一个目录：depth_est_color, 将伪深度图像都保存进去

    # scan11: 收音机、 scan13：箱子、scan33: 兔子、 scan75: 水果、 scan77: 路灯
    # scan9/15(****)/23/24/29: 建筑 (9/15/24常用)
    # input_dir = "./outputs_test0423_depthpriorN5_trans_cpc_adaptCost_64/scan77"  # for DTU,每次测试只需要将这个路径改变即可
    input_dir = "./outputs_test0509NoTransFine_tnt_config/adv/Palace"   # for TT
    # input_dir = "./dt_depth/scan33"   # 可视化GT真值
    depth_input_dir = os.path.join(input_dir, "depth_est")   # ./outputs/scanX/depth_est/
    # depth_input_dir = os.path.join(input_dir, "depth_gt")   # 可视化GT真值深度
    # depth_img_save_dir = os.path.join(input_dir, "depth_est_color")  
    depth_img_save_dir = os.path.join(input_dir, "depth_gt_color")  
    if not os.path.exists(depth_img_save_dir):  # ./outputs/scanX/depth_est_color/
        os.makedirs(depth_img_save_dir)
    for filename in os.listdir(depth_input_dir):  # ./outputs/scanX/depth_est/
        if filename.endswith(".pfm"):

            depth_path = os.path.join(depth_input_dir, filename)  # ./outputs/scanX/depth_est/0000000X.pfm
            depth_img_save_name = filename.split(".")[0]  # 0000000X
            # 伪彩色深度图像保存路径     # ./outputs/scanX/depth_est_color/0000000X.png
            depth_img_save_path = os.path.join(depth_img_save_dir, "{}.png".format(depth_img_save_name))

            # 读取pfm深度,并转换为伪深度jpg格式进行保存
            depth, scale = read_pfm(depth_path)
            # print("******* the estimae of depth is: ", depth)
            mi_d = np.min(depth[depth > 0])
            ma_d = np.max(depth)
            depth = (depth - mi_d) / (ma_d - mi_d + 1e-8)  # 归一化
            depth = (255 * depth).astype(np.uint8)  # 映射到0~255
            im_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)  # 转换为伪彩色图
            cv2.imwrite(depth_img_save_path, im_color)  # 保存
    

    # 读取场景各视角下与重建的深度图一同输出的置信度：scanX/confidence，将pfm格式转换为png格式
    # 在confidence同一级目录创建confidence
    # inputs_conf = "./outputs_test1119_cpcloss/scan77/confidence"
    # in_path = os.path.join(inputs_conf, "00000005.pfm")
    # prob_img_dir = os.path.join(inputs_conf, "prob_img")
    # if not os.path.exists(prob_img_dir):
    #     os.makedirs(prob_img_dir)
    # depth_conf, scale2 = read_pfm(in_path)
    # prob_img_save_path = os.path.join(prob_img_dir, "000000005.png")

    # conf = (255 * depth_conf).astype(np.uint8)  # 映射到0~255, 概率图都是0-1之间的值，所以不需要进行归一化
    # prob_img = cv2.applyColorMap(conf, cv2.COLORMAP_JET)  # 转换为伪彩色图
    # cv2.imwrite(prob_img_save_path, prob_img)  # 保存
    # print(depth_conf)
    # print(depth_conf.shape)
    
    # conf_error = torch.from_numpy(np.ascontiguousarray(depth_conf)) > 0.7
    # print(torch.mean(conf_error.float()))

    # inputs_depth = "./outputs_test1119_cpcloss/scan33/depth_est"
    # depth_path = os.path.join(inputs_depth, "00000000.pfm")
    # depth, scale2 = read_pfm(depth_path)
    # print(depth)
    # print(depth.shape)



    """
    以下都是测试代码，可以忽视
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
    # image_path = "./outputs_test0103_depthpriorN5_adaptCost/scan77/images/00000001.jpg"
    # pil_image = Image.open(image_path)

    # # # Convert PIL image to NumPy array
    # # # normail 0~255 to 0~1
    # numpy_image = np.array(pil_image, dtype=np.float32)/255.
    # # # numpy_image = np.array(pil_image, dtype=np.float32)

    # # # Convert the image to grayscale if it's a color image
    # if len(numpy_image.shape) == 3:
    #     gray_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
    # else:
    #     gray_image = numpy_image

    # # # Apply Sobel operator for edge detection
    # sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

    # # # Compute the magnitude of the gradient
    # magnitude = np.sqrt(sobel_x**2 + sobel_y**2) * 255.

    # # # # 使用opencv保存图像进行查看比plt要方便
    # cv2.imwrite("./combined_sobel_scan77_8.jpg", magnitude)
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

