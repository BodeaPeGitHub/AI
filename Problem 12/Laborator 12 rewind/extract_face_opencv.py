import cv2
from glob import glob
import os
import dlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neural_network
from sklearn.metrics import confusion_matrix

detector = dlib.get_frontal_face_detector()
width, height = 150, 150
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def crop_image(img, bbox, width, height):
    x, y, w, h = bbox
    img_cropped = img[y:h, x:w]
    img_cropped = cv2.resize(img_cropped, (width, height))
    return img_cropped


def extract_faces(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        cv2.resize(gray, (width, height))
        return gray
    face = faces[0]
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cropped_face = crop_image(gray, (x1, y1, x2, y2), width, height)
    return cropped_face


def flatten(image):
    img = []
    for i in image:
        for j in i:
            img.append(j)
    return img


def test():
    face = extract_faces('data/test.jpg')
    print(face)
    plt.imshow(face)
    plt.show()


def load_data():
    path = 'data\\faces\\train\\'
    train_input = []
    train_output = []
    for emotion in emotions:
        new_path = path + emotion + '\\*'
        for file in glob(new_path)[:4000]:
            train_input.append(flatten(extract_faces(file)))
            train_output.append(emotions.index(emotion))
    test_input = []
    test_output = []
    path = 'data\\faces\\test\\'
    for emotion in emotions:
        new_path = path + emotion + '\\*'
        for file in glob(new_path)[:100]:
            test_input.append(flatten(extract_faces(file)))
            test_output.append(emotions.index(emotion))
    return train_input, train_output, test_input, test_output


def plotConfusionMatrix(cm, class_names, title=""):
    import itertools
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                 horizontalalignment='center',
                 color='white' if cm[row, column] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def evalMultiClass(real_labels, computed_labels, labels_names):
    real_labels = list(real_labels)
    computed_labels = list(computed_labels)
    confusion_mat = confusion_matrix(real_labels, computed_labels)
    acc = sum([confusion_mat[i][i] for i in range(len(labels_names))]) / len(real_labels)
    prec = {}
    rec = {}
    for i in range(len(labels_names)):
        prec[labels_names[i]] = confusion_mat[i][i] / sum([confusion_mat[j][i] for j in range(len(labels_names))])
        rec[labels_names[i]] = confusion_mat[i][i] / sum([confusion_mat[i][j] for j in range(len(labels_names))])
    return acc, prec, rec, confusion_mat


if __name__ == '__main__':
    train_input, train_output, test_input, test_output = load_data()
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(12, 12), max_iter=10000)
    classifier.fit(train_input, train_output)
    computed_outputs = classifier.predict(test_input)
    print('Accuracy:', classifier.score(test_input, test_output))
    accuracy, precision, recall, matrix = evalMultiClass(np.array(test_output), computed_outputs, emotions)
    plotConfusionMatrix(matrix, emotions, "Aaa")
    # image = [[img for i in range(150)] for img in test_input[:3]]
    # result_c = computed_outputs[:3]
    # result_t = test_output[:3]
    # for img, c, r in zip(image, result_c, result_t):
    #     plt.imshow(img)
    #     plt.title("Computed: " + str(emotions[c]) + '\nReal: ' + str(emotions[r]))
    #     plt.show()
