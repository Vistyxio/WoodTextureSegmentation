import cv2 as cv
import numpy as np


# 遍历每一张图片
for i in range(12):
    # 读取图片
    img = cv.imread("./pic/{}.bmp".format(i + 1))
    # 通道分离
    B, G, R = cv.split(img)
    # 通道相减+阈值化
    # Python直接通道相减有负数会溢出 <0直接变成255
    _, GB = cv.threshold(G - B, 100, 255, cv.THRESH_BINARY)
    # 闭运算填充木板以外的区域
    cl = cv.morphologyEx(GB, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (311, 31)))
    # 反色
    cl = 255 - cl
    # 轮廓查找
    contours, hierarchy = cv.findContours(cl, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 最小外接矩形
    box = cv.boxPoints(cv.minAreaRect(contours[0]))
    # 查找左上角点和右下角点
    LU0 = int(min(box[0][0], box[1][0], box[2][0], box[3][0]))
    LU1 = int(min(box[0][1], box[1][1], box[2][1], box[3][1]))
    RD0 = int(max(box[0][0], box[1][0], box[2][0], box[3][0]))
    RD1 = int(max(box[0][1], box[1][1], box[2][1], box[3][1]))
    # ROI裁剪
    wood = img[LU1 : RD1, LU0 : RD0].copy()

    # 对ROI进行处理
    gray = cv.cvtColor(wood, cv.COLOR_BGR2GRAY)
    # gaus = cv.GaussianBlur(gray, (3, 3), -1, -1)
    # adth = cv.adaptiveThreshold(gaus, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, -1)
    # adthcl = cv.morphologyEx(adth, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (7, 7)))
    # adthop = cv.morphologyEx(adthcl, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))

    clahe = cv.createCLAHE(40.0, (12, 12))
    clahe_img = clahe.apply(gray)
    # medb = cv.medianBlur(clahe_img, 3)
    # medb = cv.GaussianBlur(clahe_img, (17, 17), -1, -1)
    medb = cv.bilateralFilter(clahe_img, 9, 300, 300)
    # adth = cv.adaptiveThreshold(medb, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, -10)

    cv.imshow("gray", gray)
    cv.imshow("clahe_img", clahe_img)
    cv.imshow("medb", medb)
    # cv.imshow("adth", adth)


    edges = cv.Canny(medb, 140, 150)

    cv.imshow("Canny", edges)


    x = cv.Sobel(medb, cv.CV_16S, 1, 0)
    y = cv.Sobel(medb, cv.CV_16S, 0, 1)
    Scale_absX = cv.convertScaleAbs(x)  # 格式转换函数
    Scale_absY = cv.convertScaleAbs(y)
    result = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)  # 图像混合

    cv.imshow("Sobel", result)

    sobth = cv.adaptiveThreshold(result, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, -12)

    cv.imshow("Sobel th", sobth)

    bwand = cv.bitwise_and(edges, sobth)
    cv.imshow("bitwise_and", bwand)
    
    # adthcl = cv.morphologyEx(bwand, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (7, 7)))
    # adthop = cv.morphologyEx(adthcl, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))



    # dil = cv.dilate(bwand, (3, 3), iterations=4)
    bwandcl = cv.morphologyEx(bwand, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))


    # cv.imshow("dil", dil)


    contours2, hierarchy2 = cv.findContours(bwandcl, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for ind, _ in enumerate(contours2):
        cona = cv.contourArea(contours2[ind])
        mina = cv.minAreaRect(contours2[ind])
        (width, height) = mina[1]
        rotrec = width * height
        if cona > 25 and rotrec / cona > 12 and max(width, height) / min(width, height) > 1.2:
            cv.drawContours(wood, contours2, ind, (255, 0, 0), 1, cv.LINE_AA)


    







    # 绘制各关键步骤图像
    cv.imshow("G-B{}".format(i + 1), G - B)
    cv.imshow("GB{}".format(i + 1), GB)
    cv.imshow("cl", cl)
    cv.imshow("wood{}".format(i + 1), wood)
    # 绘制木板角点
    for eachpoint in box:
        cv.circle(img, (int(eachpoint[0]), int(eachpoint[1])), 3, (0, 255, 0), 2)
    # 绘制各关键步骤图像
    cv.imshow("img{}".format(i + 1), img)
    cv.imshow("gray{}".format(i + 1), gray)
    # cv.imshow("gaus{}".format(i + 1), gaus)
    # cv.imshow("adth{}".format(i + 1), adth)
    cv.imshow("bwandcl{}".format(i + 1), bwandcl)
    # cv.imshow("adthop{}".format(i + 1), adthop)

    # 保持显示图片 如果按下ESC(ASCII为27)则退出
    if cv.waitKey(0) == 27:
        break
    # 销毁本次循环的所有图像窗口
    cv.destroyAllWindows()
