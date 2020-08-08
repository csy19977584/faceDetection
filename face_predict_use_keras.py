#-*- coding: utf-8 -*-

import cv2
from train2 import Model
#from face_train import Model

if __name__ == '__main__':
   
        
    model = Model()
    
    model.load_model(file_path = 'L:/dabian/model/dabian11.face.model.h5')
                   
    color = (0, 255, 0)
    
    #
    cap = cv2.VideoCapture(1)
    
    cascade_path = "C:/OpenCV/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml"

    while True:
        ret, frame = cap.read()  # 读取一帧视频

        if ret is True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cascade = cv2.CascadeClassifier(cascade_path)

            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect


                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    faceID = model.face_predict(image)


                    if faceID == 0:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        cv2.putText(frame, 'wangxuege',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽
                    if faceID == 1:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        cv2.putText(frame, 'chengshiyi',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽

                    if faceID == 2:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        cv2.putText(frame, 'linying',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽

                    if faceID == 3:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        cv2.putText(frame, 'wangyu',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线

                    if faceID == 4:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        cv2.putText(frame, 'other',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽

                    if faceID == 5:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        cv2.putText(frame, 'test',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽

        cv2.imshow("recognize", frame)


        k = cv2.waitKey(10)

        if k & 0xFF == ord('q'):
            break

        # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
