import numpy as np
import cv2
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
import skimage.color as color

abc_freq = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=10, learning_rate=1)


def subsampling_2_one(matrix):
    H, W = matrix.shape
    newH = int(H / 2)
    newW = int(W / 2)
    out = np.zeros((newH, newW))
    for i in range(newH):
        for j in range(newW):
            out[i, j] = np.floor(matrix[2 * i:2 * i + 2, 2 * j:2 * j + 2].mean())
    return out.astype(np.uint8)


def subsampling_2_three(matrix):
    H, W, C = matrix.shape
    newH = int(H / 2)
    newW = int(W / 2)
    out = np.zeros((newH, newW, C))
    for i in range(newH):
        for j in range(newW):
            for k in range(C):
                out[i, j, k] = np.floor(matrix[2 * i:2 * i + 2, 2 * j:2 * j + 2, k].mean())
    return out.astype(np.uint8)


def resizer(matrix):
    H, W = matrix.shape
    newH = 2 * H
    newW = 2 * W
    out = np.zeros((newH, newW))
    for i in range(H):
        for j in range(W):
            value = matrix[i, j]
            out[2 * i:2 * i + 2, 2 * j:2 * j + 2] = value
    return out


def improcess(filename, code):  # LAB 이미지 추출

    if code == 3:
        rgbimg = filename
        w, h, _ = rgbimg.shape
        labimg = color.rgb2lab(cv2.GaussianBlur((rgbimg), (5, 5), 2.5))  ########################################
    return labimg[:, :, 0]


def feature_ext(labpatch):
    labpatch = np.asarray(labpatch)

    ps, _ = labpatch.shape

    ### DFT
    F_l = np.log(1.000 + abs(np.fft.fftshift(np.fft.fft2(labpatch))))
    newF = F_l[int(ps / 2):ps, int(ps / 2):ps]

    newF[int(ps / 2 - 1), 0] = 0.000
    frequency_vector = newF.reshape(int(ps * ps / 4), )
    return frequency_vector


def vectmaker(img, pix, std):  # 이미지에서 Frequency Vector 추출
    sz = img.shape
    lh = sz[0]
    lw = sz[1]
    rw = int((lw - pix) / std + 1)
    rh = int((lh - pix) / std + 1)

    labimg = improcess(img, 3)

    freq_normmat = np.zeros([rh * rw, 2])

    mean_freq_vector = np.zeros([int((pix * pix / 4)), ])

    freq_vectmat = np.zeros([rh * rw, int((pix * pix / 4))])

    ### Feature vector extraction
    kk = 0
    for k in range(0, lh - pix + 1, std):
        for j in range(0, lw - pix + 1, std):
            freq_vectmat[kk, :] = feature_ext(labimg[k:pix + k, j:pix + j])
            mean_freq_vector = mean_freq_vector + freq_vectmat[kk, :]
            kk += 1
    mean_freq_vector = mean_freq_vector / kk

    ### Euclidean distance & Cosine Similarity
    for qq in range(kk):  # 위치마다 Mean Vector와의 차이 크기와 Mean Vector와 이루는 코사인 값 추출
        freq_normmat[qq, 0] = np.float32(np.linalg.norm(freq_vectmat[[qq]] - mean_freq_vector))
        freq_normmat[qq, 1] = np.float32(np.inner(freq_vectmat[[qq]], mean_freq_vector) / (
                    np.linalg.norm(freq_vectmat[[qq]]) * np.linalg.norm(mean_freq_vector)))

    return freq_normmat


pix = 16


def SVM_boost_predict_test(img, std=4, threshold=3, threshold_sum=16 * 16,
                           abcf_name="./freq_only_10tree_th4_ths256_v5"):
    abcf = joblib.load("{0}.pkl".format(abcf_name))
    img = subsampling_2_three(img)
    sz = img.shape
    lh = sz[0]
    lw = sz[1]
    resultimg = np.zeros([lh, lw])
    fv = vectmaker(img, pix, std)

    cnt = 0
    for y in range(0, lh - pix + 1, std):
        for x in range(0, lw - pix + 1, std):
            if abcf.predict(fv[[cnt]]) > 0:
                resultimg[y:y + pix, x: x + pix] += 1
            cnt += 1
    resultimg = resizer(resultimg)
    resultimg = np.floor(resultimg / threshold)

    if np.sum(resultimg) > threshold_sum:
        return 1
    return 0
