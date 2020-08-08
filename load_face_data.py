import os
import sys
import numpy as np
import cv2
 
IMAGE_SIZE = 64
 
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    
    h, w, _ = image.shape
    
    longest_edge = max(h, w)    
    
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    
    BLACK = [0, 0, 0]
    
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    return cv2.resize(constant, (height, width))
 


testnum=0
def read_path(path_name):
    images = []
    labels = []
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        for dir_mini_item in os.listdir(full_path):
            mini_item = os.path.abspath(os.path.join(full_path, dir_mini_item))
            if mini_item.endswith('.jpg'):
                image = cv2.imread(mini_item)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                new = mini_item[14:][:mini_item[14:].find('\\')]
                labels.append(new)
                #dir_item[14:][:dir_item[14:].find('\\')]
    return images,labels
    
 
def load_dataset(path_name):
    images,labels = read_path(path_name)    
    
    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    images = np.array(images)
    labelList = []
    for label in labels:
        if label == 'ajiao':
            labelList.append(0)
        if label == 'chengshiyi':
            labelList.append(1)
        if label == 'yelaoshi':
            labelList.append(2)
        if label == 'wangyu':
            labelList.append(3)
        if label == 'other':
            labelList.append(4)
        if label == 'test':
            labelList.append(5)

    labels = np.array(labelList)
    print(images.shape) 
    print(len(labels))      
    return images,labels
 
if __name__ == '__main__':

    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))    
    else:
        images = load_dataset("L:\\saveimage1")
    """
    str = "chengshiyi\\4.jpg"
    print(str[:str.find('\\')])"""