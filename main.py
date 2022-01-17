import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
)


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()
        self.prepare_data()

    def initUI(self):
        button = [None] * 3
        button[0] = QPushButton("1. Draw Keypoints", self)
        button[0].clicked.connect(self.draw_keypoints)
        button[1] = QPushButton("2. Matched Keypoints", self)
        button[1].clicked.connect(self.matched_keypoints)
        button[2] = QPushButton("3. Warp Images", self)
        button[2].clicked.connect(self.warp_images)

        vbox = QVBoxLayout()
        vbox.addWidget(button[0])
        vbox.addWidget(button[1])
        vbox.addWidget(button[2])
        self.setLayout(vbox)

    def prepare_data(self):
        # =====================
        # Prepare for button 1
        # =====================
        """
        Take reference from https://www.andreasjakl.com/
        understand-and-apply-stereo-rectification-for-depth-maps-part-2/
        """
        sift = cv2.xfeatures2d.SIFT_create()
        img1 = cv2.imread("Shark1.jpg", cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread("Shark2.jpg", cv2.IMREAD_GRAYSCALE)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Sort keypoints and descriptors of img1
        kp_num = len(kp1)
        sort_idx = sorted(range(kp_num), key=lambda i: kp1[i].size, reverse=True)
        kp1_sort = []
        des1_sort = []
        for i in range(kp_num):
            kp1_sort.append(kp1[sort_idx[i]])
            des1_sort.append(list(des1[sort_idx[i]]))
        des1_sort = np.array(des1_sort)

        # Sort keypoints and descriptors of img2
        kp_num = len(kp2)
        sort_idx = sorted(range(kp_num), key=lambda i: kp2[i].size, reverse=True)
        kp2_sort = []
        des2_sort = []
        for i in range(kp_num):
            kp2_sort.append(kp2[sort_idx[i]])
            des2_sort.append(list(des2[sort_idx[i]]))
        des2_sort = np.array(des2_sort)

        # Draw first 200 keypoints
        kp1_200 = kp1_sort[:200]
        kp2_200 = kp2_sort[:200]
        des1_200 = des1_sort[:200]
        des2_200 = des2_sort[:200]
        self.sift_img1 = cv2.drawKeypoints(
            img1, kp1_200, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )
        self.sift_img2 = cv2.drawKeypoints(
            img2, kp2_200, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        # =====================
        # Prepare for button 2
        # =====================
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des2_200, des1_200, k=2)

        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                good.append(m)

        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        )
        self.matchimg = cv2.drawMatchesKnn(
            img2, kp2_200, img1, kp1_200, matches, None, **draw_params
        )

        # =====================
        # Prepare for button 3
        # =====================
        """
        Take reference from https://stackoverflow.com/questions/
        13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
        """
        pts_src = []
        pts_dst = []
        for m in good:
            pts_src.append(kp2_200[m.queryIdx].pt)
            pts_dst.append(kp1_200[m.trainIdx].pt)
        pts_src = np.array(pts_src)
        pts_dst = np.array(pts_dst)

        H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
        h1, w1 = img1.shape
        self.warp = cv2.warpPerspective(img2, H, (w1 * 2, h1))
        self.warp[0:h1, 0:w1] = img1

    def draw_keypoints(self):
        cv2.imshow("SIFT 1", self.sift_img1)
        cv2.imshow("SIFT 2", self.sift_img2)

    def matched_keypoints(self):
        cv2.imshow("Matched Keypoints", self.matchimg)

    def warp_images(self):
        cv2.imshow("Warp Image", self.warp)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
