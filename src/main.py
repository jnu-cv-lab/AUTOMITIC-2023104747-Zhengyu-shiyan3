import cv2
import numpy as np
import matplotlib
# 关键修复：强制使用能显示窗口的后端
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# -------------------------- 工具函数定义 --------------------------
def compute_mse_psnr(original, restored):
    mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr

def fft2_shift_log(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude

def dct2d(img):
    return cv2.dct(np.float32(img))

def compute_low_freq_energy_ratio(dct_img, ratio=0.1):
    h, w = dct_img.shape
    h_low = int(h * ratio)
    w_low = int(w * ratio)
    low_freq = dct_img[:h_low, :w_low]
    total_energy = np.sum(np.abs(dct_img) ** 2)
    low_energy = np.sum(np.abs(low_freq) ** 2)
    return low_energy / total_energy

# -------------------------- 1. 图像读入 --------------------------
img_path = "/home/lenovo/cv-course/label3/src/屏幕截图 2026-03-29 130850.png"  # 改成你自己的图片路径！
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if original is None:
    raise FileNotFoundError(f"找不到图片：{img_path}，请把图片放在代码同一文件夹！")

h, w = original.shape
scale = 0.5
new_h, new_w = int(h * scale), int(w * scale)

# -------------------------- 2. 下采样 --------------------------
down_no_filter = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
blurred = cv2.GaussianBlur(original, (5, 5), 0)
down_gauss = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

# -------------------------- 3. 图像恢复 --------------------------
interp_methods = {
    "最近邻内插": cv2.INTER_NEAREST,
    "双线性内插": cv2.INTER_LINEAR,
    "双三次内插": cv2.INTER_CUBIC
}

restored_no_filter = {}
restored_gauss = {}

for name, method in interp_methods.items():
    restored_no_filter[name] = cv2.resize(down_no_filter, (w, h), interpolation=method)
    restored_gauss[name] = cv2.resize(down_gauss, (w, h), interpolation=method)

# -------------------------- 4. 打印结果 --------------------------
print("="*60)
print("【无预滤波直接缩小后恢复】")
print("="*60)
for name, img in restored_no_filter.items():
    mse, psnr = compute_mse_psnr(original, img)
    print(f"{name}: MSE = {mse:.2f}, PSNR = {psnr:.2f} dB")

print("\n" + "="*60)
print("【高斯平滑后缩小再恢复】")
print("="*60)
for name, img in restored_gauss.items():
    mse, psnr = compute_mse_psnr(original, img)
    print(f"{name}: MSE = {mse:.2f}, PSNR = {psnr:.2f} dB")

# -------------------------- 5. 显示第一张图（空间域对比）--------------------------
plt.figure(figsize=(18, 12))
plt.subplot(3, 4, 1)
plt.imshow(original, cmap="gray")
plt.title("原图")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.imshow(down_no_filter, cmap="gray")
plt.title("无预滤波缩小")
plt.axis("off")

plt.subplot(3, 4, 3)
plt.imshow(down_gauss, cmap="gray")
plt.title("高斯平滑后缩小")
plt.axis("off")

for idx, (name, img) in enumerate(restored_no_filter.items(), start=4):
    plt.subplot(3, 4, idx)
    plt.imshow(img, cmap="gray")
    mse, psnr = compute_mse_psnr(original, img)
    plt.title(f"{name}\nMSE={mse:.1f}")
    plt.axis("off")

for idx, (name, img) in enumerate(restored_gauss.items(), start=8):
    plt.subplot(3, 4, idx)
    plt.imshow(img, cmap="gray")
    mse, psnr = compute_mse_psnr(original, img)
    plt.title(f"{name}\nMSE={mse:.1f}")
    plt.axis("off")

plt.tight_layout()
plt.show()  # 强制显示

# -------------------------- 6. 显示第二张图（傅里叶）--------------------------
bilinear_restored_no_filter = restored_no_filter["双线性内插"]
fft_original = fft2_shift_log(original)
fft_down_no_filter = fft2_shift_log(down_no_filter)
fft_bilinear_restored = fft2_shift_log(bilinear_restored_no_filter)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(fft_original, cmap="gray")
plt.title("原图傅里叶谱")
plt.axis("off")

plt.subplot(132)
plt.imshow(fft_down_no_filter, cmap="gray")
plt.title("缩小后")
plt.axis("off")

plt.subplot(133)
plt.imshow(fft_bilinear_restored, cmap="gray")
plt.title("恢复后")
plt.axis("off")

plt.show()

# -------------------------- 7. 显示第三张图（DCT）--------------------------
dct_original = dct2d(original)
dct_bilinear_restored = dct2d(bilinear_restored_no_filter)

energy_original = compute_low_freq_energy_ratio(dct_original, 0.1)
energy_bilinear = compute_low_freq_energy_ratio(dct_bilinear_restored, 0.1)

print("\nDCT低频能量占比：")
print(f"原图：{energy_original:.4f}")
print(f"双线性恢复：{energy_bilinear:.4f}")

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(20*np.log(np.abs(dct_original)+1), cmap="gray")
plt.title("原图DCT")
plt.axis("off")

plt.subplot(122)
plt.imshow(20*np.log(np.abs(dct_bilinear_restored)+1), cmap="gray")
plt.title("恢复DCT")
plt.axis("off")

plt.show()