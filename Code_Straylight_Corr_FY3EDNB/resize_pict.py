import cv2
import os

def resize_image(image_path):
    # read image
    gray_matrix = cv2.imread(image_path)
    gray_image = cv2.cvtColor(gray_matrix, cv2.COLOR_BGR2GRAY)
    
    # Adjust image size to specified pixels
    resized_image = cv2.resize(gray_image , (1522, 2000), interpolation=cv2.INTER_LINEAR)

    # save image
    output_path=image_path
    cv2.imwrite(output_path, resized_image)


# dir_path=r'/share/home/dq113/liangye2020/python_picture_after/outer_hdf5_samples/20220613_1620'
# for i in os.listdir(dir_path):
#     path=os.path.join(dir_path,i)
#     ig=cv2.imread(path)
#     gray_ig = cv2.cvtColor(ig, cv2.COLOR_BGR2GRAY)
#     print(i,gray_ig.shape)
    





