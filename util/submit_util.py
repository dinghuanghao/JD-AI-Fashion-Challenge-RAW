import os
import numpy as np
from util import path

def save_submit(predict, name="submit.txt"):
    images = []
    with open(path.TEST_DATA_TXT, "r") as f:
        for i in f.readlines():
            images.append(i.split(",")[0])

    predict = predict.astype(np.int8)
    with open(os.path.join(path.SUBMIT_RESULT_PATH, name), "w+") as f:
        for i in range(len(images)):
            s = ",".join([str(j) for j in predict[i]])
            f.write("%s,%s\n" % (images[i], s))

def build_test_submit(predict_file, file_name, threshold=0.2):
    images = []
    with open(path.TEST_DATA_TXT, "r") as f:
        for i in f.readlines():
            images.append(i.split(",")[0])

    predict = np.load(predict_file)
    # predict = predict > threshold
    # predict = predict.astype(np.int8)
    with open(os.path.join(path.SUBMIT_RESULT_PATH, file_name + ".txt"), "w+") as f:
        for i in range(len(images)):
            s = ",".join([str(j) for j in predict[i]])
            f.write("%s,%s\n" % (images[i], s))


if __name__ == '__main__':
    build_test_submit(os.path.join(path.CNN_RESULT_PATH,
                              r"model-xception-record-model11-val2-['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']weights.010.hdf5.predict.npy"),
                      "model-xception-record-model11-val2-['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']weights.010.hdf5.predict.npy",
                      threshold=0.120000)
