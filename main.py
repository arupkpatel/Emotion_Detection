from keras.models import load_model
classifier = load_model('cdi.h5')

import numpy as np
from keras.preprocessing import image
import cv2

img = cv2.imread('1.jpg',0)
img2=cv2.imread('1.jpg')
img = cv2.resize(img,(48,48))
# print(img.shape)
img = image.img_to_array(img)
# print(img.shape)
img = np.expand_dims(img, axis=0)
result = classifier.predict(img)
# train_set.class_indices
if result[0][0] == 1:
    print('sad')
    cv2.putText(img2,'sad',(20, 100),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0), thickness=4)

else:
    print('happy')
    cv2.putText(img2, 'happy',(20, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=4)

cv2.imshow('Result', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()