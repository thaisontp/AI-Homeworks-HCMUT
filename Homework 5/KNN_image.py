import os
import cv2
import numpy as np

# --- 2 DÒNG QUAN TRỌNG ĐỂ ÉP BẬT CỬA SỔ TRÊN WINDOWS ---
import matplotlib

matplotlib.use('TkAgg')  # Sử dụng bộ dựng giao diện hệ thống trực tiếp
# ------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def compress_image(image_path, k):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape

    X = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = np.uint8(kmeans.cluster_centers_)

    compressed_X = centers[labels]
    compressed_img = compressed_X.reshape(original_shape)

    return img, compressed_img


def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def evaluate_k_values(image_path, k_list=[2, 4, 8, 16, 32]):
    # Đã sửa lỗi cú pháp nrows và ncols chuẩn xác
    fig, axes = plt.subplots(nrows=1, ncols=len(k_list) + 1, figsize=(20, 5))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i, k in enumerate(k_list):
        _, compressed_img = compress_image(image_path, k)
        psnr_val = calculate_psnr(img, compressed_img)

        axes[i + 1].imshow(compressed_img)
        axes[i + 1].set_title(f"K={k}\nPSNR: {psnr_val:.2f}dB")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    sample_image = "1.jpg"

    if os.path.exists(sample_image):
        print(f"Evaluating compression performance on: {sample_image}")
        evaluate_k_values(sample_image, k_list=[2, 4, 8, 16, 32])
    else:
        print(f"Loi: Khong tim thay file '{sample_image}'.")
