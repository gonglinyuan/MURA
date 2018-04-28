import time

import train


def main():
    # runTest()
    run_train()


def run_train():
    timestamp = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")

    path_data_train = '../../MURA_trainval_keras'
    path_data_valid = '../../MURA_valid1_keras'

    model_name = 'DENSE-NET-121'
    model_pretrained = True

    batch_size = 16
    epoch_num = 100

    img_size = 256
    crop_size = 224

    path_model = 'm-' + timestamp + '.pth.tar'

    print('Training NN architecture = ', model_name)
    train.train(path_data_train, path_data_valid, model_name, model_pretrained, batch_size, epoch_num, img_size,
                crop_size, timestamp, None)

    print('Testing the trained model')
    train.test(path_data_valid, path_model, model_name, model_pretrained, batch_size, img_size, crop_size, timestamp)


def runTest():
    pathDirData = './database'
    pathFileTest = './dataset/test_1.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = './models/m-25012018-123527.pth.tar'

    timestampLaunch = ''

    train.test(pathDirData, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
               imgtransCrop, timestampLaunch)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
