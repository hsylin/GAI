import os
import cv2

if __name__ == '__main__':
    # 確保目錄存在
    output_dir = 'data/denoising'
    os.makedirs(output_dir, exist_ok=True)

    # 讀取圖像
    image = cv2.imread('data/denoising/3.png')
    
    # 將圖像大小調整為150x150
    img = cv2.resize(image, (64,64))
    
    # 保存調整後的圖像
    output_path = os.path.join(output_dir, '3.png')
    cv2.imwrite(output_path, img)
