from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2


class classification():
    def __init__(self):
        self.label = ''
        self.classify()

    def classify(self):
        caltech_dir = "./multi_img_data/imgs_others_test_sketch"

        image_w = 128   # image 의 width
        image_h = 128   # image 의 height

        X = []
        filenames = []
        files = glob.glob(caltech_dir+"/*.*")

        data_size = 0
        img = ''

        for i, f in enumerate(files):
            data_size += 1
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))

            data = np.asarray(img)
            filenames.append(f)

            X.append(data)
        
        plt.imshow(img)             # Resized image 를 출력
        plt.title("Resized Image")
        plt.show()

        X = np.array(X)
        print("X shape :", X.shape)
        model = load_model('./model/multi_img_classification_6_64_relu.model')
        prediction = model.predict(X)

        print("PREDICTION BEFORE\n", prediction)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})    # label 들에 대한 확률값들을 소수 셋째자리까지 반올림한다.

        print("PREDICTION AFTER\n", prediction)
        print("----------------------------------")
        # sg = Segmentation()

        # print("START PREDICTION")
        for i, v in enumerate(prediction):
            if i != data_size - 1: # 방금 Sketch 한 그림만 checking 하도록 함
                continue;
            
            pre_ans = v.argmax()  # 예측레이블
            pre_ans_str = ''

            if pre_ans == 0: pre_ans_str = "사과"       # 빨강, 초록
            elif pre_ans == 1: pre_ans_str = "당근"     # 주황
            elif pre_ans == 2: pre_ans_str = "참외"     # 노랑
            elif pre_ans == 3: pre_ans_str = "딸기"     # 빨강, 초록
            elif pre_ans == 4: pre_ans_str = "토마토"   # 빨강, 초록
            elif pre_ans == 5: pre_ans_str = "수박"     # 초록
            else: pre_ans_str = "식별 불가능"

            # 0.8 이상의 percentage 를 갖고 있는 label이
            #   label value 로 여겨진다.
            label_percentage = 0.8

            if v[0] >= label_percentage:
                self.label = 'apple'
            elif v[1] >= label_percentage:
                self.label = 'carrot'
            elif v[2] >= label_percentage:
                self.label = 'orientalmelon'
            elif v[3] >= label_percentage:
                self.label = 'strawberry'
            elif v[4] >= label_percentage:
                self.label = 'tomato'
            elif v[5] >= label_percentage:
                self.label = 'watermelon'
            else:   # 가장 큰 확률의 label 이 0.8을 넘지 못 하면 없는 data로 출력한다.
                print("해당 이미지는 없는 데이터입니다.")
                self.label = 'none'