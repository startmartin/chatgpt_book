import base64
import json
import os
from time import time

import cv2
import zxing
import imutils
import numpy as np
from imutils import auto_canny
from imutils.perspective import four_point_transform
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from loguru import logger
from pyzbar import pyzbar
from pyzbar.wrapper import ZBarSymbol
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models

from common.recognition_exception import RecognitionException
from common.util import generate_random_str, upload_image


class Recognition:
    QUESTION_TYPE_SINGLE = 1  # 单选
    QUESTION_TYPE_MULTI = 2  # 多选
    QUESTION_TYPE_BLANK = 3  # 单填空
    QUESTION_TYPE_MULTI_BLANK = 4  # 多填空
    QUESTION_TYPE_SA = 5  # 简答题
    QUESTION_TYPE_JUDGE = 6  # 判断题

    def __init__(self, config):
        self.image = cv2.imread(config['url'])
        self.paper = None  # 调试用
        self.origin_paper = None
        self.origin_paper_r = None
        self.origin_warped = None  # 原图大小，识别二维码用
        self.origin_warped_r = None
        self.config = config
        self.select_area_min = 62  # 选择题填充面积最小值，正常160左右
        self.judge_area_min = 200  # 判断题填充面积最小值，正常360左右
        # 图像缩放比例
        self.scaling_ratio = 0
        self.jump_judge = False

    def handle(self):
        start_time = time()
        image = self.image
        if image is None:
            logger.error('读取图片失败 {}', config['url'])
            raise RecognitionException(-3001, '读取图像失败')
        # url = os.getenv('OSS_DOMAIN') + self.config['url']
        # image = imutils.url_to_image(url)
        logger.debug('读取图片耗时 {}', time() - start_time)

        result = {'answerList': {}, 'picList': {}}
        try:
            paper, warped, paper_r, warped_r = self.preprocess_image(image)
            self.paper = paper

            if self.config.get('ignoreQrCodeConfig', False):
                page_no = self.config['ignoreQrCodeConfig']['pageNo']
            else:
                # 识别二维码
                qr_data = self.get_qr_data(self.origin_warped)
                if not qr_data:
                    logger.info('换个方向识别二维码')
                    self.origin_paper = self.origin_paper_r
                    self.origin_warped = self.origin_warped_r
                    qr_data = self.get_qr_data(self.origin_warped)
                    if not qr_data:
                        raise RecognitionException(-3000, '二维码识别失败')
                    else:
                        warped = warped_r
                        self.paper = paper_r
                logger.debug('qr data：{}；'.format(qr_data))
                qr_data = qr_data.split('-')  # id-页码
                result['qrData'] = qr_data[0]
                page_no = int(qr_data[1])
        except Exception as e:
            logger.warning(e)
            raise RecognitionException(-3000, '二维码识别失败')

        if self.jump_judge:
            logger.debug('Return results directly：{}；'.format(result))
            return result

        # 对灰度图应用大津二值化算法
        _, thresh = cv2.threshold(
            warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # thresh = cv2.threshold(warped, 210, 255, cv2.THRESH_BINARY_INV)[1]

        question_list = []
        for page in self.config['pageList']:
            if page['pageNo'] == page_no:
                question_list = page['questionList']
        self.get_fill_answer(thresh, question_list, result)
        if not result['studentNo'] and not self.config.get('ignoreStudentNo', False):
            raise RecognitionException(-3001, '学号识别失败',
                                       {'qrData': result.get('qrData', {})})
        self.get_rect_answer(question_list, result)

        # plt.figure("paper")
        # plt.imshow(imutils.opencv2matplotlib(self.paper))
        # plt.show()

        logger.debug('Final return result：{}；'.format(result))
        return result

    def get_fill_answer(self, image, question_list, result):
        # 识别学号
        student_no_config = self.config['studentNo']
        if self.config.get('ignoreStudentNo', False):
            student_no = ''
        elif student_no_config.get('code128', False):
            student_no = self.get_code128_data(self.origin_warped)
        else:
            cnts = cv2.findContours(
                image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            student_no_area = None

            for c in cnts:
                # 计算轮廓的边界框，然后利用边界框数据计算宽高比
                (x, y, w, h) = cv2.boundingRect(c)
                # if 30 > w > 10 and 40 < x < 350 and 5 < h < 25:
                #     self.draw_contours(self.paper, c)
                #     print('区域 debug', x, y, w, h)

                # 学号部分
                if student_no_config['outer']['w'] - 30 <= w <= student_no_config['outer']['w'] + 30 \
                    and student_no_config['outer']['h'] - 30 <= h <= student_no_config['outer']['h'] + 30 \
                    and student_no_config['outer']['leftTop'][0] - 50 <= x <= student_no_config['outer']['leftTop'][
                        0] + 50 \
                    and student_no_config['outer']['leftTop'][1] - 50 <= y <= student_no_config['outer']['leftTop'][
                        1] + 50:
                    student_no_area = self.crop_image(image, x, y, w, h)
                    logger.debug('识别到学号区域 {} {} {} {}'.format(x, y, w, h))
                    # self.draw_contours(self.paper, c)
                    break

            if student_no_area is None:
                # 识别不到轮廓就直接截图尝试识别
                x = student_no_config['outer']['leftTop'][0] - 10
                y = student_no_config['outer']['leftTop'][1] - 10
                w = student_no_config['outer']['w'] + 15
                h = student_no_config['outer']['h'] + 15
                student_no_area = self.crop_image(image, x, y, w, h)
                logger.debug('识别不到学号区域，直接根据坐标截取：', x, y, w, h)
                # plt.imshow(imutils.opencv2matplotlib(student_no_area))
            student_no = self.get_student_no(student_no_area)

        result['studentNo'] = student_no

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # dilated = cv2.dilate(image, kernel)

        # 识别选择题
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        # plt.figure("opened")
        # plt.imshow(imutils.opencv2matplotlib(opened))
        # plt.show()
        # exit()

        cnts = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # 对每一个轮廓进行循环处理
        for c in cnts:
            # 计算轮廓的边界框，然后利用边界框数据计算宽高比
            (x, y, w, h) = cv2.boundingRect(c)
            for question in question_list:

                if question['type'] in (
                        self.QUESTION_TYPE_SINGLE,
                        self.QUESTION_TYPE_MULTI,
                        self.QUESTION_TYPE_JUDGE):
                    # 选择题和判断题部分
                    c_x = x + w / 2
                    c_y = y + h / 2

                    single_option_width = question['w'] / \
                        question['optionsNum']  # 单个选项宽度
                    if question['leftTop'][0] - 10 <= c_x <= question['leftTop'][0] + question['w'] + 10 \
                            and question['leftTop'][1] - 7 <= c_y <= question['leftTop'][1] + question['h'] + 7 \
                            and single_option_width - 12 <= w <= single_option_width + 12:
                        # 判断填涂区域是否足够
                        rate, pix = self.get_white_rate(
                            self.crop_image(image, x, y, w, h))
                        if rate < 60:
                            logger.debug(
                                '填充比例不通过：题号 {}；{} {} {} {} {} {}'.format(question['id'], x, y, w, h, rate,
                                                                         pix))
                            continue
                        # 判断区域面积是否足够
                        area = cv2.contourArea(c)
                        if (question['type'] == self.QUESTION_TYPE_JUDGE and area < self.judge_area_min) \
                                or (question['type'] != self.QUESTION_TYPE_JUDGE and area < self.select_area_min):
                            logger.debug(
                                '区域面积不通过：题号 {}；{} {} {} {} {} {}'.format(question['id'], x, y, w, h, area,
                                                                         pix))
                            continue

                        no = (c_x - question['leftTop']
                              [0]) // single_option_width
                        option = self.get_option(int(no))

                        if question['type'] == self.QUESTION_TYPE_JUDGE:
                            option = True if no == 0 else False

                        if question['id'] not in result['answerList']:
                            result['answerList'][question['id']] = [option]
                        else:
                            result['answerList'][question['id']].append(option)

                        logger.debug('识别到选择/判断题：题号 {}；坐标 {} {} {} {}；答案：{}；填充比例：{}；面积：{}'.format(
                            question['id'], x, y, w, h, option, rate, area))
                        # self.draw_contours(self.paper, c)

    def get_rect_answer(self, question_list, result):
        for question in question_list:
            x = int((question['leftTop'][0] - 10) * self.scaling_ratio)
            y = int((question['leftTop'][1] - 10) * self.scaling_ratio)
            w = int((question['w'] + 15) * self.scaling_ratio)
            h = int((question['h'] + 15) * self.scaling_ratio)
            answer_area = self.crop_image(self.origin_paper, x, y, w, h)
            key = generate_random_str()
            # key = upload_image(self.config['uploadDir'] + generate_random_str() + '.jpg',
            #                    cv2.imencode(".jpg", answer_area)[1].tobytes())
            logger.debug('答题截图 {} {}', question['id'], key)
            result['picList'][question['id']] = key

            if question['type'] in (
                    self.QUESTION_TYPE_BLANK,
                    self.QUESTION_TYPE_MULTI_BLANK,
                    self.QUESTION_TYPE_SA):
                logger.debug('识别填空/简答题 {} {}', question['id'], key)
                result['answerList'][question['id']] = key

            if question['id'] == 23:
                plt.figure(question['id'])
                plt.imshow(imutils.opencv2matplotlib(answer_area))
                plt.show()

    def get_white_rate(self, image):
        area = 0
        height, width = image.shape
        for i in range(height):
            for j in range(width):
                if image[i, j] == 255:
                    area += 1
        return area * 100 / (height * width), area

    def preprocess_image(self, image):
        start_time = time()
        paper, warped = self.transformation(image)
        height = self.config['h']
        width = self.config['w']
        paper_r = warped_r = None
        # 判断是否忽略二维码，忽略的话就不需要根据二维码判断方向了
        if self.config.get('ignoreQrCodeConfig', False):
            rotate_code = self.ignore_qrcode_rotate()
            if rotate_code is not None:
                paper = cv2.rotate(paper, rotate_code)
                warped = cv2.rotate(warped, rotate_code)
        else:
            # 处理试卷不是正面时的情况
            paper, warped, paper_r, warped_r = self.rotate(paper, warped)

        # 保存调整分辨率前的图
        self.origin_paper = paper
        self.origin_paper_r = paper_r
        self.origin_warped = warped
        self.origin_warped_r = warped_r
        # 调整分辨率
        paper = imutils.resize(paper, height=height, width=width)
        warped = imutils.resize(warped, height=height, width=width)
        if paper_r is not None:
            paper_r = imutils.resize(paper_r, height=height, width=width)
        if warped_r is not None:
            warped_r = imutils.resize(warped_r, height=height, width=width)

        self.scaling_ratio = self.origin_warped.shape[0] / warped.shape[0]
        logger.debug('缩放比例 {} ', self.scaling_ratio)
        logger.debug('预处理图片耗时 {} ', time() - start_time)

        return paper, warped, paper_r, warped_r

    def transformation(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel = np.ones((5, 5), np.uint8)
        blurred = cv2.erode(blurred, kernel, iterations=1)  # 腐蚀
        blurred = cv2.dilate(blurred, kernel, iterations=2)  # 膨胀

        # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        plt.figure("寻找定位坐标")
        plt.imshow(imutils.opencv2matplotlib(morph))
        plt.show()

        # edged = auto_canny(morph)
        cnts, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        print("原始图片检测的轮廓总数：", len(cnts))
        if len(cnts) < 4:
            raise RecognitionException(-3000, '未找到定位点')

        candidate_contours = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            # print("轮廓周长：", peri)

            # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # print('原始轮廓的边数:', len(c), ', 近似轮廓的边数:', len(approx))

            # 当近似轮廓为4时，代表是需要提取的矩形区域
            if len(approx) == 4:
                # self.draw_contours(edged, c)
                candidate_contours.append(c)

        location_points = []
        for c in candidate_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            location_points.append([x, y])

        # 对定位点排序，排序后顺序为：左上、右上、右下、左下
        location_points = self.order_points(np.array(location_points))

        pts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if location_points[0][0] == x and location_points[0][1] == y:  # 左上
                pts.append([x, y])
                continue
            if location_points[1][0] == x and location_points[1][1] == y:  # 右上
                pts.append([x + w, y])
                continue
            if location_points[2][0] == x and location_points[2][1] == y:  # 右下
                pts.append([x + w, y + h])
                continue
            if location_points[3][0] == x and location_points[3][1] == y:  # 左下
                pts.append([x, y + h])
                continue

        logger.debug('定位点：{}'.format(pts))
        pts = np.float32(pts)

        paper = four_point_transform(image, pts.reshape(4, 2))
        warped = four_point_transform(gray, pts.reshape(4, 2))
        # plt.figure("warped")
        # plt.imshow(imutils.opencv2matplotlib(warped))
        # plt.figure("paper")
        # plt.imshow(imutils.opencv2matplotlib(paper))
        # plt.show()
        # exit(0)
        return paper, warped

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="int")  # 按照左上、右上、右下、左下顺序初始化坐标

        s = pts.sum(axis=1)  # 计算点xy的和
        rect[0] = pts[np.argmin(s)]  # 左上角的点的和最小
        rect[2] = pts[np.argmax(s)]  # 右下角的点的和最大

        diff = np.diff(pts, axis=1)  # 计算点xy之间的差
        rect[1] = pts[np.argmin(diff)]  # 右上角的差最小
        rect[3] = pts[np.argmax(diff)]  # 左下角的差最小
        return rect  # 返回4个顶点的顺序

    def rotate(self, paper, warped):
        h, w, _ = paper.shape
        if self.config['h'] > self.config['w']:
            if h > w:
                paper_r = cv2.rotate(paper, cv2.ROTATE_180)
                warped_r = cv2.rotate(warped, cv2.ROTATE_180)
            else:
                paper = cv2.rotate(paper, cv2.ROTATE_90_CLOCKWISE)
                paper_r = cv2.rotate(paper, cv2.ROTATE_180)
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                warped_r = cv2.rotate(warped, cv2.ROTATE_180)
        else:
            if h < w:
                paper_r = cv2.rotate(paper, cv2.ROTATE_180)
                warped_r = cv2.rotate(warped, cv2.ROTATE_180)
            else:
                paper = cv2.rotate(paper, cv2.ROTATE_90_CLOCKWISE)
                paper_r = cv2.rotate(paper, cv2.ROTATE_180)
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                warped_r = cv2.rotate(warped, cv2.ROTATE_180)

        return paper, warped, paper_r, warped_r

    def ignore_qrcode_rotate(self):
        # 旋转方向
        rotate_code = None

        ignore_qrcode_config = self.config['ignoreQrCodeConfig']
        outer = ignore_qrcode_config['outer']
        inner = ignore_qrcode_config['inner']

        outer_h = self.point_distance_point(
            outer['leftTop'], outer['leftBottom'])
        outer_w = self.point_distance_point(
            outer['leftTop'], outer['rightTop'])

        if self.config['h'] > self.config['w']:
            # A4答题卡
            if outer_h > outer_w:
                # 计算点到上下两条边的距离，短的为顶边
                distance1 = self.point_distance_line(
                    inner['leftTop'], outer['leftTop'], outer['rightTop'])
                distance2 = self.point_distance_line(
                    inner['leftBottom'], outer['leftBottom'], outer['rightBottom'])
                if distance1 > distance2:
                    rotate_code = cv2.ROTATE_180
            else:
                # 计算点到左右两条边的距离，短的为顶边
                distance1 = self.point_distance_line(
                    inner['leftTop'], outer['leftTop'], outer['leftBottom'])
                distance2 = self.point_distance_line(
                    inner['rightTop'], outer['rightTop'], outer['rightBottom'])
                if distance1 > distance2:
                    rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
                else:
                    rotate_code = cv2.ROTATE_90_CLOCKWISE
        else:
            # A3答题卡
            if outer_h < outer_w:
                # 计算点到上下两条边的距离，短的为顶边
                distance1 = self.point_distance_line(
                    inner['leftTop'], outer['leftTop'], outer['rightTop'])
                distance2 = self.point_distance_line(
                    inner['leftBottom'], outer['leftBottom'], outer['rightBottom'])
                if distance1 > distance2:
                    rotate_code = cv2.ROTATE_180
            else:
                # 计算点到左右两条边的距离，短的为顶边
                distance1 = self.point_distance_line(
                    inner['leftTop'], outer['leftTop'], outer['leftBottom'])
                distance2 = self.point_distance_line(
                    inner['rightTop'], outer['rightTop'], outer['rightBottom'])
                if distance1 > distance2:
                    rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
                else:
                    rotate_code = cv2.ROTATE_90_CLOCKWISE

        return rotate_code

    # 点到点的距离
    def point_distance_point(self, point1, point2):
        vec1 = np.array(point1)
        vec2 = np.array(point2)
        distance = np.linalg.norm(vec1 - vec2)
        return distance

    # 点到直线距离
    def point_distance_line(self, point, line_point1, line_point2):
        point = np.array(point)
        line_point1 = np.array(line_point1)
        line_point2 = np.array(line_point2)
        # 计算向量
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / \
            np.linalg.norm(line_point1 - line_point2)
        return distance

    def get_qr_data(self, image):
        qr_code_config = self.config['qrCode']
        qr_image = self.crop_image(image, int(qr_code_config['leftTop'][0] * self.scaling_ratio) - 20,
                                   int(qr_code_config['leftTop'][1]
                                       * self.scaling_ratio) - 15,
                                   int(qr_code_config['w'] * self.scaling_ratio) +
                                   int(30 * self.scaling_ratio),
                                   int(qr_code_config['h'] * self.scaling_ratio) + int(30 * self.scaling_ratio))
        # plt.imshow(imutils.opencv2matplotlib(qr_image))
        # plt.show()
        # exit()

        # zbar 识别
        barcodes = pyzbar.decode(qr_image, symbols=[ZBarSymbol.QRCODE])
        if barcodes:
            logger.debug('zbar 识别二维码成功')
            data = barcodes[0].data.decode("utf-8")
            return str(data)

        # zxing 识别
        filename = generate_random_str(16) + '.jpg'
        cv2.imwrite(filename, qr_image)
        reader = zxing.BarCodeReader()
        data = reader.decode(filename, possible_formats=['QR_CODE'])
        os.remove(filename)
        if data:
            logger.debug('zxing 识别二维码成功')
            data = data.parsed
            return str(data)

        # 在otsu二值结果的基础上，不断增加阈值，用于识别模糊图像
        threshold, bin_img = cv2.threshold(
            qr_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        barcodes = self.get_qr_in_bin_img(bin_img)
        if barcodes:
            logger.debug('阈值 {} 识别二维码成功'.format(threshold))
            data = barcodes[0].data.decode("utf-8")
            return str(data)

        # 如果阈值otsuthreshold失败，则采用高斯自适应阈值化，可以识别出一定的控制阈值也识别不出来的二维码
        bin_img = cv2.adaptiveThreshold(
            qr_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 0)
        barcodes = self.get_qr_in_bin_img(bin_img)
        if barcodes:
            logger.debug('高斯自适应阈值化 识别二维码成功')
            data = barcodes[0].data.decode("utf-8")
            return str(data)

        start_time = time()
        logger.debug('初始化阈值 {}'.format(threshold))
        while threshold < 255:
            bin_img = cv2.threshold(
                qr_image, threshold, 255, cv2.THRESH_BINARY)[1]
            barcodes = self.get_qr_in_bin_img(bin_img)
            if barcodes:
                logger.debug('阈值 {} 识别二维码成功'.format(threshold))
                # plt.imshow(imutils.opencv2matplotlib(qr_image1))
                # plt.show()
                data = barcodes[0].data.decode("utf-8")
                return str(data)
            threshold += 5  # 步长越大，识别率越低，速度越快
        logger.debug('阈值识别耗时 {}', time() - start_time)

        # 使用腾讯云服务进行识别
        start_time = time()
        data = self.decode_by_tencent(qr_image)
        logger.debug('腾讯云识别耗时 {}', time() - start_time)
        if data:
            return str(data)

        logger.debug('识别二维码失败')

        return {}

    def get_code128_data(self, image):
        code_128_config = self.config['studentNo']['code128']
        image = self.crop_image(image, int(code_128_config['leftTop'][0] * self.scaling_ratio) - 20,
                                int(code_128_config['leftTop']
                                    [1] * self.scaling_ratio) - 15,
                                int(code_128_config['w'] *
                                    self.scaling_ratio) + 30,
                                int(code_128_config['h'] * self.scaling_ratio) + 30)
        data = ''
        # zbar 识别
        barcodes = pyzbar.decode(image, symbols=[ZBarSymbol.CODE128])
        if barcodes:
            logger.debug('zbar 识别一维码成功')
            data = barcodes[0].data.decode("utf-8")

        # 使用腾讯云服务进行识别
        if not data:
            data = self.decode_by_tencent(image)

        logger.debug('学号识别结果：{}', data)

        return data

    def decode_by_tencent(self, image):
        try:
            cred = credential.Credential(
                os.getenv('TENCENTCLOUD_SECRET_ID'), os.getenv('TENCENTCLOUD_SECRET_KEY'))

            http_profile = HttpProfile()
            http_profile.endpoint = "ocr.tencentcloudapi.com"

            client_profile = ClientProfile()
            client_profile.httpProfile = http_profile
            client = ocr_client.OcrClient(cred, "ap-guangzhou", client_profile)

            req = models.QrcodeOCRRequest()
            req.ImageBase64 = self.image_to_base64(image)
            resp = json.loads(client.QrcodeOCR(req).to_json_string())
            data = resp['CodeResults'][0]['Url']
            logger.debug('腾讯云 识别成功')

            return data
        except TencentCloudSDKException as err:
            logger.debug(err)
            return ''

    def image_to_base64(self, image_np):
        image = cv2.imencode('.jpg', image_np)[1]
        image_code = str(base64.b64encode(image))[2:-1]

        return image_code

    # 对二值图像进行识别，如果失败则开运算进行二次识别
    def get_qr_in_bin_img(self, bin_img):
        barcodes = pyzbar.decode(bin_img, symbols=[ZBarSymbol.QRCODE])
        if not barcodes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
            barcodes = pyzbar.decode(opened, symbols=[ZBarSymbol.QRCODE])
        return barcodes

    def get_student_no(self, image):
        logger.debug('---开始识别学号---')
        student_no_config = self.config['studentNo']
        x_param = student_no_config['outer']['w'] // 5
        # plt.figure('studentNo')
        # plt.imshow(imutils.opencv2matplotlib(image))
        # plt.show()
        # exit()

        area_coordinate_list = []
        # 先找出轮廓
        mask = np.zeros(image.shape, dtype=np.uint8)
        thresh = image

        # 水平线检测
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        detect_horizontal = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        detect_horizontal = cv2.dilate(
            detect_horizontal, horizontal_kernel, iterations=3)
        cnts = cv2.findContours(
            detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(mask, [c], -1, (255, 255, 255), 3)

        # 垂直线检测
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        detect_vertical = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        detect_vertical = cv2.dilate(
            detect_vertical, vertical_kernel, iterations=3)  # 连接竖直线
        cnts = cv2.findContours(
            detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(mask, [c], -1, (255, 255, 255), 3)

        # plt.figure('mask')
        # plt.imshow(imutils.opencv2matplotlib(mask))
        # plt.show()
        # exit()
        cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            # 根据列切割学号
            if x_param - 7 <= w <= x_param + 5 and \
                    student_no_config['inner']['h'] - 10 <= h <= student_no_config['inner']['h'] + 5:
                area_coordinate_list.append([c, x, y, w, h])

        # print(len(area_coordinate_list))
        # exit()
        student_no_list = {}
        student_no_column_list = []
        if len(area_coordinate_list) == 5:
            for area_coordinate in area_coordinate_list:
                student_no_column = self.crop_image(
                    image, *area_coordinate[1:])
                student_no_column_list.append(
                    [area_coordinate[1], student_no_column])
                # self.judge_student_no_column(student_no_column, area_coordinate[1], student_no_list)
        self.judge_student_no_column_by_row(
            student_no_column_list, student_no_list)

        student_no_list = [str(student_no_list[k])
                           for k in sorted(student_no_list.keys())]
        student_no = ''.join(student_no_list)
        logger.debug('识别学号结束：{}'.format(student_no))
        if len(student_no) != 5:
            student_no = ''

        return student_no

    def judge_student_no_column(self, student_no_column, i, student_no_list):
        student_no_config = self.config['studentNo']
        y_param = student_no_config['inner']['h'] // 10

        # 开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        student_no_column = cv2.morphologyEx(
            student_no_column, cv2.MORPH_OPEN, kernel, iterations=2)

        cnts = cv2.findContours(
            student_no_column, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for j, c in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            rate, pix = self.get_white_rate(
                self.crop_image(student_no_column, x, y, w, h))
            area = cv2.contourArea(c)
            # print(x, y, w, h, rate, area)
            if (rate > 45 or 500 > pix > 100) and area > self.select_area_min and 30 > x + w / 2 > 10 \
                    and x != 0 and y != 0:
                # student_no = int((y + h / 2) // y_param)
                student_no = int((y + 7) // y_param)
                student_no_list[(i + 1) * 100 + j] = student_no
                # print(student_no, x, y, w, h, y_param, rate, pix)
                # self.draw_contours(student_no_column, c)
        # plt.figure(self.generate_random_str())
        # plt.imshow(imutils.opencv2matplotlib(student_no_column))
        # plt.show()

    def judge_student_no_column_by_row(self, student_no_column_list, student_no_list):
        for i, student_no_column in student_no_column_list:
            # 腐蚀图像
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            student_no_column = cv2.erode(
                student_no_column, kernel, iterations=2)
            # 缩放到指定高度，是为了能平均分为10份
            student_no_column = imutils.resize(student_no_column, height=180)
            h, w = student_no_column.shape
            crop_h = h // 10
            avg_pix = 0
            all_pix = []
            for j in range(10):
                item_image = self.crop_image(
                    student_no_column, 0, j * crop_h, w, crop_h)
                _, pix = self.get_white_rate(item_image)
                avg_pix += pix
                all_pix.append(pix)

                # print(i, j, avg_pix, pix)
                # plt.figure(self.generate_random_str())
                # plt.imshow(imutils.opencv2matplotlib(item_image))
                # plt.show()
            avg_pix = avg_pix / 10
            for j in range(10):
                # 暂时定为大于平均值+10的为填充区域，实际上其他区域经过erode处理后白色像素点为0
                if all_pix[j] > avg_pix + 10:
                    student_no_list[(i + 1) * 100 + j] = j
                    logger.debug('根据像素点判断学号 {} {} {} {}',
                                 i, j, avg_pix, all_pix[j])

            # plt.figure(self.generate_random_str())
            # plt.imshow(imutils.opencv2matplotlib(student_no_column))
            # plt.show()

    def crop_image(self, image, x, y, w, h):
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return image[y1:y2, x1:x2]

    def get_center_point(self, c):
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy

    def get_option(self, no):
        options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        return options[no]

    # 标记识别出来的轮廓，调试用
    def draw_contours(self, image, c):
        M = cv2.moments(c)
        if M["m00"] != 0:
            c_x = int(M["m10"] / M["m00"])
            c_y = int(M["m01"] / M["m00"])
            # 绘制中心及其轮廓
            cv2.drawContours(image, c, -1, (0, 0, 255), 5, lineType=0)
            cv2.circle(image, (c_x, c_y), 7, (255, 255, 255), -1)


if __name__ == '__main__':
    load_dotenv()
    # type 1 单选,2 多选,3 单填,4 多填,5 简答,6 判断
    with open('./test.json', 'r') as f:
        config = json.loads(f.read())
    print(json.dumps(config))
    start_time = time()
    a = Recognition(config)
    result = a.handle()
    print(json.dumps(result))
    end_time = time()
    print('cost time: ', end_time - start_time)
