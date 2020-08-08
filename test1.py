import cv2
import sys
cap=cv2.VideoCapture(1) #打开设备索引号对于设备的摄像头，一般电脑的默认索引号为0
classfier = cv2.CascadeClassifier("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")
color = (0, 255, 0)
num=0
path_name='L:/saveimage1/chengshiyi'
path_name1 = 'L:/saveimage2/chengshiyi'
while cap.isOpened():
    ret,frame=cap.read()
    if ret == True:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:            #大于0则检测到人脸                                   
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect
                if num <= 100:
                    img_name = '%s/%d.jpg' % (path_name, num)
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    cv2.imwrite(img_name, image)
                if num > 100:
                    img_name1 = '%s/%d.jpg' % (path_name1, num)
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    cv2.imwrite(img_name1, image)
                if num>105:
                    break
                num = num + 1
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX#字体
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)
                #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        if num > 105:
            break
        cv2.imshow("video",frame)
        if cv2.waitKey(20)&0xFF==ord('q'):
            break
    else:
        break
 
cap.release()
cv2.destroyAllWindows()