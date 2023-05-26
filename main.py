import cv2
import numpy as np
import openpyxl
import os


def psnr(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    psnr_value = cv2.PSNR(image1, image2)
    return psnr_value


def H263(image):
    image = image.astype(np.float64)
    deblocking_image = np.copy(image)
    m, n = deblocking_image.shape
    block_size = 4
    QP = 130

    for i in range(block_size, m - block_size + 1, block_size):
        for j in range(block_size, n - block_size + 1, block_size):
            for k in range(block_size):
                # Vertical deblocking
                A = deblocking_image[i + k, j - 2]
                B = deblocking_image[i + k, j - 1]
                C = deblocking_image[i + k, j]
                D = deblocking_image[i + k, j + 1]
                d = (3 * A - 8 * B + 8 * C - 3 * D) / 16
                d1 = np.sign(d) * (np.maximum(0, np.abs(d) - np.maximum(0, 2 * np.abs(d) - QP)))
                deblocking_image[i + k, j - 1] = B + d1
                deblocking_image[i + k, j] = C - d1

                # Horizontal deblocking
                A = deblocking_image[i - 2, j + k]
                B = deblocking_image[i - 1, j + k]
                C = deblocking_image[i, j + k]
                D = deblocking_image[i + 1, j + k]
                d = (3 * A - 8 * B + 8 * C - 3 * D) / 16
                d1 = np.sign(d) * (np.maximum(0, np.abs(d) - np.maximum(0, 2 * np.abs(d) - QP)))
                deblocking_image[i - 1, j + k] = B + d1
                deblocking_image[i, j + k] = C - d1

    return deblocking_image


def H264(image):
    image = image.astype(np.float64)
    deblocking_image = np.copy(image)
    m, n = deblocking_image.shape
    block_size = 4
    alpha = 50
    beta = 0

    for i in range(block_size, m - block_size + 1, block_size):
        for j in range(block_size, n - block_size + 1, block_size):
            for k in range(block_size):
                p3 = deblocking_image[i, j - 4]
                p2 = deblocking_image[i, j - 3]
                p1 = deblocking_image[i, j - 2]
                p0 = deblocking_image[i, j - 1]
                q0 = deblocking_image[i, j]
                q1 = deblocking_image[i, j + 1]
                q2 = deblocking_image[i, j + 2]
                q3 = deblocking_image[i, j + 3]
                ap = np.abs(p2 - p0)
                aq = np.abs(q2 - q0)

                # Left/upper side
                if ap < beta and np.abs(p0 - q0) < ((alpha >> 2) + 2):
                    deblocking_image[i, j - 1] = np.uint8((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) // 8)
                    deblocking_image[i, j - 2] = np.uint8((p2 + p1 + p0 + q0 + 2) // 4)
                    deblocking_image[i, j - 3] = np.uint8((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) // 8)
                else:
                    deblocking_image[i, j - 1] = np.uint8((2 * p1 + p0 + q1 + 2) // 4)

                if aq < beta and np.abs(p0 - q0) < ((alpha >> 2) + 2):
                    deblocking_image[i, j] = np.uint8((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) // 8)
                    deblocking_image[i, j + 1] = np.uint8((p0 + q0 + q1 + q2 + 2) // 4)
                    deblocking_image[i, j + 2] = np.uint8((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) // 8)
                else:
                    deblocking_image[i, j] = np.uint8((2 * q1 + q0 + p1 + 2) // 4)
    return deblocking_image


frame_number = 28
picture_name = []
Decompressed = np.zeros((frame_number,), dtype=object)
Original = np.zeros((frame_number,), dtype=object)
Decompressed_psnr = np.zeros((frame_number,))
H263_deblock_img = np.zeros((frame_number,), dtype=object)
H263_psnr = np.zeros((frame_number,))
H264_deblock_img = np.zeros((frame_number,), dtype=object)
H264_psnr = np.zeros((frame_number,))

for i in range(frame_number):
    filename = f"./Hw3_test sequences/Holmes/Decompressed/({i + 1}).bmp"
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Decompressed[i] = image
    filename = f"./Hw3_test sequences/Holmes/Original/({i + 1}).jpg"
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Original[i] = image
    Decompressed_psnr[i] = psnr(Original[i], Decompressed[i])
    picture_name.append(str(i + 1))

output_dir = 'H263_deblock_image'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(frame_number):
    H263_deblock_img[i] = H263(Decompressed[i])
    filename = f"./H263_deblock_image/({i + 1}).bmp"
    cv2.imwrite(filename, H263_deblock_img[i])
    H263_psnr[i] = psnr(Original[i], H263_deblock_img[i])


output_dir = 'H264_deblock_image'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(frame_number):
    H264_deblock_img[i] = H264(Decompressed[i])
    filename = f"./H264_deblock_image/({i+1}).bmp"
    cv2.imwrite(filename, H264_deblock_img[i])
    H264_psnr[i] = psnr(Original[i], H264_deblock_img[i])

psnr_data = np.empty((4, frame_number + 1), dtype=object)
psnr_data[0, 0] = 'Picture Name'
psnr_data[1, 0] = 'Decompressed PSNR'
psnr_data[2, 0] = 'H.263 PSNR'
psnr_data[3, 0] = 'H.264 PSNR'

for i in range(frame_number):
    psnr_data[0, i + 1] = picture_name[i]
    psnr_data[1, i + 1] = Decompressed_psnr[i]
    psnr_data[2, i + 1] = H263_psnr[i]
    psnr_data[3, i + 1] = H264_psnr[i]

wb = openpyxl.Workbook()
ws = wb.active

for r in range(psnr_data.shape[0]):
    for c in range(psnr_data.shape[1]):
        ws.cell(row=r + 1, column=c + 1, value=psnr_data[r, c])

wb.save('psnr.xlsx')
