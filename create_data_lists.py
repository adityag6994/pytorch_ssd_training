from utils import create_data_lists, create_data_lists_rafeeq

if __name__ == '__main__':
    # COCO
    # create_data_lists(voc07_path='/home/agupta/Desktop/naukri2021/pytorch_detection_tensorrt/pytorch_detection_tensorrt/a-PyTorch-Tutorial-to-Object-Detection/data/mergeVOC2007/VOC2007',
    #                   voc12_path='/home/agupta/Desktop/naukri2021/pytorch_detection_tensorrt/pytorch_detection_tensorrt/a-PyTorch-Tutorial-to-Object-Detection/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',
    #                   output_folder='./')

    # rafeeq
    create_data_lists_rafeeq(
        rafeeq_path='data/rafeeq',
        output_folder='data/rafeeq/')
