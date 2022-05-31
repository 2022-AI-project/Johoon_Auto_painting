import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from datetime import datetime
from classification import classification
from Fill_color import Fill_color
import copy
import cv2

from PyQt5.QtWidgets import QMessageBox

class drawing_board(QWidget):
    def __init__(self):
        super().__init__()
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # 화면크기스케일링
        self.file = ''
        self.file2 = ''
        self.file3 = ''

        # 전체 폼 박스
        formbox = QHBoxLayout()
        self.setLayout(formbox)

        # 좌, 우 레이아웃박스
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.right2 = QVBoxLayout()

        self.drawType = 1

        self.combo = QComboBox()
 
        for i in range(4, 21):
            self.combo.addItem(str(i))
        self.pencolor = QColor(0, 0, 0)
        self.penbtn = QPushButton()
        self.penbtn.setStyleSheet('background-color: rgb(0,0,0)')
        self.penbtn.clicked.connect(self.showColorDlg)

        self.brushcolor = QColor(255, 255, 255)                             # brush color == black
        self.brushbtn = QPushButton()                                       
        self.brushbtn.setStyleSheet('background-color: rgb(255,255,255)')
        self.brushbtn.clicked.connect(self.showColorDlg)

        self.checkbox = QCheckBox('지우개 동작')
        self.checkbox.stateChanged.connect(self.checkClicked)

        # 우 레이아웃 박스에 그래픽 뷰 추가
        self.view = CView(self)
        self.view.setFixedWidth(256)
        self.view.setFixedHeight(256)
        left.addWidget(self.view)

        # 전체 지우기
        removebutton = QPushButton('전체 지우기', self)
        left.addWidget(removebutton)
        removebutton.clicked.connect(self.remove_all)

        # painting 버튼
        paintingbutton = QPushButton('자동 채색', self)
        left.addWidget(paintingbutton)
        paintingbutton.clicked.connect(self.load_image)

        left.addStretch(1)  # 그냥 레이아웃 여백 추가

        # 제일 오른쪽 레이아웃에 빈 흰색 배경
        pixmap = QPixmap('whiteimage.png')
        self.lbl_img = QLabel()
        self.lbl_img.setPixmap(pixmap)
        self.right2.addWidget(self.lbl_img)

        # 전체 폼박스에 레이아웃 박스 배치
        formbox.addLayout(left)
        formbox.addLayout(right)
        formbox.addLayout(self.right2)

        formbox.setStretchFactor(left, 0)
        formbox.setStretchFactor(right, 1)

        self.setGeometry(100, 100, 700, 400)

    def radioClicked(self):
        for i in range(len(self.radiobtns)):
            if self.radiobtns[i].isChecked():
                self.drawType = i
                break

    def checkClicked(self):
        pass

    def showColorDlg(self):

        # 색상 대화상자 생성
        color = QColorDialog.getColor()

        sender = self.sender()

        # 색상이 유효한 값이면 참, QFrame에 색 적용
        if sender == self.penbtn and color.isValid():
            self.pencolor = color
            self.penbtn.setStyleSheet('background-color: {}'.format(color.name()))
        else:
            self.brushcolor = color
            self.brushbtn.setStyleSheet('background-color: {}'.format(color.name()))

    def save_image(self):       # Sketch 된 image 를 저장한다.
        date = datetime.now()
        filename = 'Screenshot ' + date.strftime('%Y-%m-%d_%H-%M-%S.png')   # datetime 을 뒤에 덧붙여 저장한다.
        img = QPixmap(self.view.grab(self.view.sceneRect().toRect()))
        self.file = "./multi_img_data/imgs_others_test_sketch/" + filename  # sketch된 그림이 저장되는 path 이다.
        img.save(self.file, 'png')      # *.png 형태로 image 가 저장된다.
        img = cv2.imread(self.file, 0) 
        img = img[2:476, 2:663]         # 사진을 저장할 때 끝부분 모서리 부분을 지운다.
        cv2.imwrite(self.file, img)

    def remove_all(self):
        for i in self.view.scene.items():
            self.view.scene.removeItem(i)

    def load_image(self):
        self.save_image()               # Sketched image 를 저장

        label = classification().label  # 이미지 분류
        # label 이 정의되었다. 즉, model 을 통해 Sketched image 가 무엇인지 prediction 했다.
        # print(label)

        korean_label = ''
        if label == "apple":
            korean_label = "사과"
        elif label == "carrot":
            korean_label = "당근"
        elif label == "orientalmelon":
            korean_label = "참외"
        elif label == "strawberry":
            korean_label = "딸기"
        elif label == "tomato":
            korean_label = "토마토"
        elif label == "watermelon":
            korean_label = "수박"

        reply = self.msg_box('\"'+ korean_label +'\"을 그린 게 맞다면 \"OK\" 버튼을, 다시 그리고 싶다면 \"NO\" 버튼을 눌러주세요.')
        if reply == True:
            self.lbl_img.hide()  # 전 이미지 숨김
            if label == "apple" or label == "tomato" or label == "strawberry":  # label이 사과, 토마토, 딸기인 경우 실행된다.
                self.file2 = copy.copy(self.file)                               # 2가지 색을 칠하는 작업이 진행된다.
                fill = Fill_color(self.file, label, 1)
                self.file = fill.file

                pixmap = QPixmap(self.file)

                self.lbl_img = QLabel()
                self.lbl_img.setPixmap(pixmap)
                self.right2.addWidget(self.lbl_img)

                fill = Fill_color(self.file2, label, 2)
                self.file2 = fill.file
                # print(self.file)

                pixmap = QPixmap(self.file2)

                self.lbl_img = QLabel()
                self.lbl_img.setPixmap(pixmap)
                self.right2.addWidget(self.lbl_img)
            else:                                                               # label이 당근, 수박, 참외인 경우 실행된다.
                fill = Fill_color(self.file, label, 1)                          # 1가지 색을 칠하는 작업이 진행된다.
                self.file = fill.file
            # print(self.file)

                pixmap = QPixmap(self.file)

                self.lbl_img = QLabel()
                self.lbl_img.setPixmap(pixmap)
                self.right2.addWidget(self.lbl_img)

    def msg_box(self, text):
        msgBox = QMessageBox()
        msgBox.setText(text)
        reply = msgBox.question(self, 'message', text, QMessageBox.Ok | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.No:
            return False
        else: return True


# QGraphicsView display QGraphicsScene
class CView(QGraphicsView):

    def __init__(self, parent):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.items = []

        self.start = QPointF()
        self.end = QPointF()

        self.setRenderHint(QPainter.HighQualityAntialiasing)

    def moveEvent(self, e):
        rect = QRectF(self.rect())
        rect.adjust(0, 0, -4, -4)

        self.scene.setSceneRect(rect)

    def mousePressEvent(self, e):

        if e.button() == Qt.LeftButton:
            # 시작점 저장
            self.start = e.pos()
            self.end = e.pos()

    def mouseMoveEvent(self, e):
        # e.buttons()는 정수형 값을 리턴, e.button()은 move시 Qt.Nobutton 리턴
        if e.buttons() & Qt.LeftButton:

            self.end = e.pos()

            if self.parent().checkbox.isChecked():
                pen = QPen(QColor(255, 255, 255), 10)
                path = QPainterPath()
                path.moveTo(self.start)
                path.lineTo(self.end)
                self.scene.addPath(path, pen)
                self.start = e.pos()
                return None

            pen = QPen(self.parent().pencolor, self.parent().combo.currentIndex()+3)

            # 직선 그리기
            if self.parent().drawType == 0:

                # 장면에 그려진 이전 선을 제거
                if len(self.items) > 0:
                    self.scene.removeItem(self.items[-1])
                    del (self.items[-1])

                    # 현재 선 추가
                line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())
                self.items.append(self.scene.addLine(line, pen))

            # 곡선 그리기
            if self.parent().drawType == 1:
                # Path 이용
                path = QPainterPath()
                path.moveTo(self.start)
                path.lineTo(self.end)
                self.scene.addPath(path, pen)

                # Line 이용
                # line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())
                # self.scene.addLine(line, pen)

                # 시작점을 다시 기존 끝점으로
                self.start = e.pos()

            # 사각형 그리기
            if self.parent().drawType == 2:
                brush = QBrush(self.parent().brushcolor)

                if len(self.items) > 0:
                    self.scene.removeItem(self.items[-1])
                    del (self.items[-1])

                rect = QRectF(self.start, self.end)
                self.items.append(self.scene.addRect(rect, pen, brush))

            # 원 그리기
            if self.parent().drawType == 3:
                brush = QBrush(self.parent().brushcolor)

                if len(self.items) > 0:
                    self.scene.removeItem(self.items[-1])
                    del (self.items[-1])

                rect = QRectF(self.start, self.end)
                self.items.append(self.scene.addEllipse(rect, pen, brush))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:

            if self.parent().checkbox.isChecked():
                return None

            pen = QPen(self.parent().pencolor, self.parent().combo.currentIndex())

            if self.parent().drawType == 0:
                self.items.clear()
                line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())

                self.scene.addLine(line, pen)

            if self.parent().drawType == 2:
                brush = QBrush(self.parent().brushcolor)

                self.items.clear()
                rect = QRectF(self.start, self.end)
                self.scene.addRect(rect, pen, brush)

            if self.parent().drawType == 3:
                brush = QBrush(self.parent().brushcolor)

                self.items.clear()
                rect = QRectF(self.start, self.end)
                self.scene.addEllipse(rect, pen, brush)


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     w = drawing_board()
#     w.show()
#     sys.exit(app.exec_())