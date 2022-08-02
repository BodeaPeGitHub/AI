from keras.models import load_model

model = load_model('models/facenet_keras.h5')
# model.face_detection('a.jpg')
print(model.inputs)
print(model.outputs)
