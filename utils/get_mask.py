from segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_mask():
    image = cv2.imread("r_0.png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. 載入 SAM 模型
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").cpu()
    print("SAM model loaded successfully.")
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # 3. 進行無提示自動分割
    masks, scores, logits = predictor.predict(multimask_output=True)

    # 4. 顯示所有 masks 給你選
    for i, mask in enumerate(masks):
        plt.figure(figsize=(5, 5))
        plt.imshow(image_rgb)
        # plt.imshow(mask, cmap="jet", alpha=0.5)
        # plt.title(f"Mask {i} | Score: {scores[i]:.3f}")
        plt.axis('off')
        plt.show()

def get_mask_by_color():

    # 載入圖像（保留透明度）
    image = cv2.imread("r_0.png", cv2.IMREAD_UNCHANGED)  # RGBA
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]

    # 取得 R, G, B 通道
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # 判斷哪些 pixel 為黑色（盆栽）
    # 可根據實際畫面調整 threshold
    pot_mask = (r < 70) & (g < 70) & (b < 70) & (alpha > 0)
    pot_mask = pot_mask.astype(np.uint8) * 255  # 轉成 0 / 255

    # 儲存 mask
    cv2.imwrite("pot_mask.png", pot_mask)

    # 視覺化
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(pot_mask, cmap="gray")
    plt.title("Pot Mask")
    plt.show()


get_mask_by_color()