import cv2
import PIL
import matplotlib.pyplot as plt
from PIL import Image

# data/rafeeq/test1/images/85a3d00a-0000000.jpg 1 1 (3088, 2320)
# data/rafeeq/test1/images/cceb1e32-0000000.JPG 6 2 (3088, 2320)
# data/rafeeq/test1/images/6fcd49f0-0000000.jpg 1 3 (2320, 3088)
# data/rafeeq/test1/images/e3d4e22b-0000000.jpg 1 4 (2320, 3088)
# data/rafeeq/test1/images/afd8da90-0000000.JPG 6 6 (3088, 2320)
# data/rafeeq/test1/images/db583f88-0000000.jpg 1 8 (2320, 3088)

img_list = ['data/rafeeq/test1/images/85a3d00a-0000000.jpg',
            'data/rafeeq/test1/images/cceb1e32-0000000.JPG',
            'data/rafeeq/test1/images/6fcd49f0-0000000.jpg',
            'data/rafeeq/test1/images/e3d4e22b-0000000.jpg',
            'data/rafeeq/test1/images/afd8da90-0000000.JPG',
            'data/rafeeq/test1/images/db583f88-0000000.jpg']

# img1 = 'data/rafeeq/test1/images/6fcd49f0-0000000.jpg' #with-out EXIF
# img6 = 'data/rafeeq/test1/images/cceb1e32-0000000.JPG' #with-xxx EXIF
# imgs = [cv2.imread(img1), cv2.imread(img6)]

imgs = []
for i in img_list:
    imgs.append(cv2.imread(i))
# for i in img_list:
#     imgs.append(Image.open(i))


_, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.flatten()
count = 0
for img, ax in zip(imgs, axs):
    ax.imshow(img)
    print(img_list[count], img.shape)
    count+=1
plt.show()

# after changing
# data/rafeeq/test1/images/85a3d00a-0000000.jpg (3088, 2320)
# data/rafeeq/test1/images/cceb1e32-0000000.JPG (3088, 2320)
# data/rafeeq/test1/images/6fcd49f0-0000000.jpg (2320, 3088)
# data/rafeeq/test1/images/e3d4e22b-0000000.jpg (2320, 3088)
# data/rafeeq/test1/images/afd8da90-0000000.JPG (3088, 2320)
# data/rafeeq/test1/images/db583f88-0000000.jpg (2320, 3088)
# w/o change
# data/rafeeq/test1/images/85a3d00a-0000000.jpg (3088, 2320)
# data/rafeeq/test1/images/cceb1e32-0000000.JPG (3088, 2320)
# data/rafeeq/test1/images/6fcd49f0-0000000.jpg (2320, 3088)
# data/rafeeq/test1/images/e3d4e22b-0000000.jpg (2320, 3088)
# data/rafeeq/test1/images/afd8da90-0000000.JPG (2320, 3088)
# data/rafeeq/test1/images/db583f88-0000000.jpg (2320, 3088)
