import cv2,numpy as np 


# label_hue = np.uint8(179*faker/np.max(faker))
# blank_ch = 255*np.ones_like(label_hue)
# labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# label_hue = np.uint8(179*faker/np.max(faker))

faker = np.arange(5)+1
label_hue = np.uint8(179*faker/np.max(faker))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])


labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

print(labeled_img[1][0])