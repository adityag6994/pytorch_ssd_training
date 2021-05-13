# thanks to this guy : https://github.com/JoshVarty/CancerDetection/blob/master/01_ImageStats.ipynb
import os
import numpy
import cv2
import numpy as np
from tqdm import tqdm


def getMeans(paths):
    """calculate mean"""
    redSum = 0
    greenSum = 0
    blueSum = 0
    for path in paths:
        image = cv2.imread(path)

        red = np.reshape(image[:, :, 0], -1)
        green = np.reshape(image[:, :, 1], -1)
        blue = np.reshape(image[:, :, 2], -1)

        redMean = red.mean()
        greenMean = green.mean()
        blueMean = blue.mean()

        redSum += redMean
        greenSum += greenMean
        blueSum += blueMean

    redMean = redSum / len(paths)
    greenMean = greenSum / len(paths)
    blueMean = blueSum / len(paths)

    return redMean, greenMean, blueMean


def getStds(paths, redMean, greenMean, blueMean):
    redSum = 0
    greenSum = 0
    blueSum = 0

    for path in paths:
        image = cv2.imread(path)
        red = np.reshape(image[:, :, 0], -1)
        green = np.reshape(image[:, :, 1], -1)
        blue = np.reshape(image[:, :, 2], -1)

        redDiffs = red - redMean
        redSumOfSquares = np.sum(redDiffs ** 2)

        blueDiffs = blue - blueMean
        blueSumOfSquares = np.sum(blueDiffs ** 2)

        greenDiffs = green - greenMean
        greenSumOfSquares = np.sum(greenDiffs ** 2)

        redSum = redSum + (1 / (len(paths) * 96 * 96)) * redSumOfSquares
        greenSum = greenSum + (1 / (len(paths) * 96 * 96)) * greenSumOfSquares
        blueSum = blueSum + (1 / (len(paths) * 96 * 96)) * blueSumOfSquares

    redStd = np.sqrt(redSum)
    greenStd = np.sqrt(greenSum)
    blueStd = np.sqrt(blueSum)

    return redStd, greenStd, blueStd


train_dir = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
test_dir = 'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
dummy_dir = 'data/dummy/images/'
experiment_dir = 'data/experiments/'

train_images = os.listdir(train_dir)
test_images = os.listdir(test_dir)
dummy_images = os.listdir(dummy_dir)
experiment_images = os.listdir(experiment_dir)

train_images = [train_dir + s for s in train_images]
test_images = [test_dir + s for s in test_images]
all_images = train_images + test_images

dummy_images = [dummy_dir + s for s in dummy_images]

# array([[0.43281991, 0.43885488, 0.47329301],
#        [1.87465636, 1.56086287, 1.73204997]])
# array([[110.80189744, 112.34684868, 121.16300968],
#        [479.91202732, 399.58089571, 443.40479302]])
# dummy_images
# redMean, greenMean, blueMean = getMeans(dummy_images)
# redStd, greenStd, blueStd = getStds(dummy_images, redMean, greenMean, blueMean)

# train_images
# [[100.0, 100.0, 100.0], [255.15518153991437, 255.15518153991437, 255.15518153991437]]
# [[0.390625   0.390625   0.390625  ]
#  [0.99669993 0.99669993 0.99669993]]
redMean, greenMean, blueMean = getMeans(experiment_images)
redStd, greenStd, blueStd = getStds(experiment_images, redMean, greenMean, blueMean)

dataset_stats = [[redMean, greenMean, blueMean], [redStd, blueStd, greenStd]]
np_dataset_stats = numpy.asarray(dataset_stats)/256
print(dataset_stats)
print(np_dataset_stats)