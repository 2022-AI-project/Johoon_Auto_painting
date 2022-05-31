import cv2
import copy
import random
import math

class Fill_color(object):
    def __init__(self, filename, label, cnt):
        self.file = ''
        self.start(filename, label, cnt)

    def start(self, file_name, label, cnt): # cnt 의 값이 2인 경우는 칠해지는 색이 2종류 인 것이다.
        origin_img = cv2.imread(file_name)  # file 을 읽는다.
        if origin_img is None:
            print('================== error - not found : ' + file_name + '======================')
            return


        gray_img = cv2.imread(file_name,0)
        bin_img = self.binarize(gray_img, 250)

        print("[bin img file")
        print(bin_img)
        print("bin_img size : ", len(bin_img))
        print("bin_img[0] size : ", len(bin_img[0]))
        sg_img , count, count_size = self.segmentation(bin_img)
        result = self.segmentation_image_show(origin_img,sg_img, label, count, cnt)
        result = self.line_effect(sg_img, result, 7, 10)
        result = self.natural_coloring(result, 80)
        result = cv2.GaussianBlur(result, (3, 3), 0)
        cv2.imwrite('./multi_img_data/result/result.png', result)
        self.file = './multi_img_data/result/result.png'

    def binarize(self, img, threshold):
        # 이진화
        ret, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return bin_img

    def segmentation(self, img):
        segmentation_img = copy.deepcopy(img)
        offset = [[1, 0], [0, 1], [-1, 0], [0, -1]]     # 상하좌우로 이동하기 위한 offset 이다.
        count_size = {}                                 # 같은 색이 입혀지는 범위의 pixel 수들을 저장한다.
        count = 0
        start_point = []                                # 색칠되지 않은 pixel이 감지되는 위치이다.

        for i in range(len(img)):                       # 2d image 를
            for j in range(len(img[0])):                #   모든 pixel을 check 한다.
                if segmentation_img[i][j] == 255:       # segmentation_img[i][j] 의 값이 255 라는건 그 pixel과 인접한 영역이
                    start_point.append([i,j])           #   색칠되지 않은 상태이므로 start_point 에 넣고 작업을 시작한다.
                    count += 1                          # 한 영역이 색칠 될 것 이므로 count 값을 늘린다.

                    count_size[count] = 0               # 칠해질 영역의 count_size 를 0으로 초기화 한다.

                    q = [[i,j]]

                    while q:                            # queue 를 활용하여 dfs Algorithm 으로 현재 영역을 모두 count값으로 바꾼다.
                        cur = q.pop(0)
                        x, y = cur[0], cur[1]
                        if x < 0 or y < 0:
                            pass
                        elif x > len(img) - 1 or y > len(img[0]) - 1:   # out of bound 된 pixel 이라면
                            pass
                        elif segmentation_img[x][y] != 255:             # 이미 색이 부여된 pixel 이라면
                            pass
                        else:                                           # 색이 부여될 수 있는 pixel 이라면
                            segmentation_img[x][y] = count              #   현재 위치의 값에 count 를 부여한 후
                            count_size[count] += 1                      #   count_size[count] 값을 1 늘린다.
                            for i in range(4):                                  # 현재 pixel 에서 상하좌우 영역의 pixel을
                                q.append([x + offset[i][0], y + offset[i][1]])  #   q 에 추가한다.

        # 가장 많이 칠해진 영역이 앞에 오도록 count_size sorting 을 진행한다.
        count_size = sorted(count_size.items(), reverse = True, key = lambda item: item[1])

        return [segmentation_img, count, count_size]

    def segmentation_image_show(self,origin_img, segmentation_img , label, count, cnt):
        color_img = copy.deepcopy(origin_img)

        # label들의 korean label dictionary 를 만든다.
        dic_label = {"apple":"사과","tomato":"토마토","watermelon" : "수박","orientalmelon":"참외","strawberry":"딸기","carrot":"당근"}

        if cnt == 1:    # 여러개의 색이 filling 되는 경우 한 번만 추정 label 을 출력
            print('\n\n=== 해당 이미지는 '+dic_label[label]+'(으)로 추정됩니다 === ')
        color_count = self.return_size(copy.deepcopy(segmentation_img),20)

        if label == 'apple':                                                    # 사과
            if cnt == 1:
                color = [0, 0, 180]
            elif cnt == 2:
                color = [20, 160, 20]
            for i in range(len(segmentation_img)):
                for j in range(len(segmentation_img[0])):
                    if segmentation_img[i][j] >= 2:
                        color_img[i][j] = color
        elif label == 'tomato':                                                 # 토마토
            if cnt == 1:
                color = [[0, 0, 180], [0, 100, 0]]
            elif cnt == 2:
                color = [[20, 160, 20], [0, 100, 0]]
            for seg_cnt in range(count - 1):
                for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[seg_cnt]:
                            if seg_cnt == 0:
                                color_img[i][j] = color[0]
                            else:
                                color_img[i][j] = color[1]
        elif label == 'watermelon':                                             # 수박
            color = [10, 180, 10]
            for seg_cnt in range(count - 1):
                for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[0]:
                            color_img[i][j] = color
        elif label == 'orientalmelon':                                          # 참외
            color = [0, 255, 255]
            for seg_cnt in range(count - 1):
                for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] >= 3:
                            color_img[i][j] = color
        elif label == 'carrot':                                                 # 당근
            color = [[38, 67, 243], [10, 180, 10]]
            for seg_cnt in range(count - 1):
                for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[seg_cnt]:
                            if seg_cnt == 0:
                                color_img[i][j] = color[0]
                            else:
                                color_img[i][j] = color[1]
        elif label == 'strawberry':                                             # 딸기
            if cnt == 1:
                color = [[54,54,255],[10,180,10]]
            elif cnt == 2:
                color = [[20, 160, 20], [10, 180, 10]]
            for seg_cnt in range(count - 1):
                for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[seg_cnt]:
                            if seg_cnt == 0:
                                color_img[i][j] = color[0]
                            else:
                                color_img[i][j] = color[1]
        
        return color_img

    def return_size(self,img, return_num):
        count_list = [0] * 255
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] != 255 and img[i][j] != 0 and img[i][j] != 1:
                    count_list[img[i][j]] += 1

        count_sort_list = []
        for i in range(return_num):
            count_sort_list.append(count_list.index(max(count_list)))
            count_list[count_sort_list[i]] = 0
        return count_sort_list

    def natural_coloring(self, img, value):
        # random_num = random.randrange(125,175)
        random_num = 125
        for i in range(random_num-value,random_num+value):
            for j in range(random_num-value,random_num+value):
                d = self.p2p_dst(i,j,random_num,random_num)
                if d <= value and self.img2np(img[i][j],[0,0,0]) and self.img2np(img[i][j],[255,255,255]):
                    for k in range(0,3):
                        img[i][j][k] = self.check255(img[i][j][k] + value - d)
        return img
        # img = cv2.GaussianBlur(img, (11, 11), 0)

    def p2p_dst(self,x1,y1,x2,y2):
        return int(math.sqrt((x2-x1)**2 + (y2-y1)**2))

    def img2np(self,v1,v2):
        if v1[0] == v2[0] and v1[1] == v2[1] and v1[2] == v2[2]:
            return False
        return True

    def check255(self,v):
        if v >= 255:
            return 255
        return v

    def line_effect(self, seg_img, color_img, value, n):
        for i in range(len(seg_img)):
            for j in range(len(seg_img[0])):
                for l in range(n):
                    if i + l > 298:
                        pass
                    else:
                        if seg_img[i][j] == 0 and seg_img[i+l][j] != 0 and seg_img[i+l][j] != 1:
                            for k in range(3):
                                if color_img[i+l][j][k] - value < 0:
                                    color_img[i+l][j][k] = 0
                                else:
                                    color_img[i+l][j][k] -= value
                    if j - l <= 0:
                        pass
                    else:
                        if seg_img[i][j] == 0 and seg_img[i][j-l] != 0 and seg_img[i][j-l] != 1:
                            for k in range(3):
                                if color_img[i][j-l][k] - value < 0:
                                    color_img[i][j-l][k] = 0
                                else:
                                    color_img[i][j-l][k] -= value
        return color_img