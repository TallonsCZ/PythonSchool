import cv2 as cv

#Import pretrained model
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#Import image + convert to gray
src = cv.imread('detect.jpg')
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=4)

for (x, y, w, h) in faces:
    border_size = 20
    x_border = max(0, x - border_size)
    y_border = max(0, y - border_size)
    w_border = min(src.shape[1] - x, w + 2 * border_size)
    h_border = min(src.shape[0] - y, h + 2 * border_size)
    cv.rectangle(src, (x_border, y_border), (x_border + w_border, y_border + h_border), (255, 211, 42), 10)

#Smaller output
percent = 20
width = int(src.shape[1] * percent / 100)
height = int(src.shape[0] * percent / 100)
dim = (width, height)
new_src = cv.resize(src, dim)


cv.imshow('Output', new_src)
cv.waitKey(0)
cv.destroyAllWindows()