import cv2
import cv2.ximgproc as xip
import numpy as np

def joint_bilateral_up_sample(low_res, guide, d=5, sigma_color=0.1, sigma_space=2.0, save_img=False, output_path=""):

    low_res = np.transpose(low_res, (1,2,0))
    guide = np.transpose(guide, (1,2,0))
    guide = np.float32(guide)

    new_width = guide.shape[0]
    new_height = guide.shape[1]

    up_scaled_f = cv2.resize(low_res, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    filtered_f = xip.jointBilateralFilter(guide, up_scaled_f, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    out = np.clip(filtered_f * 255.0, 0, 255).astype(np.uint8)

    if save_img:
        cv2.imwrite(output_path, out)

    return out