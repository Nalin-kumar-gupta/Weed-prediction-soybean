from resize import *
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=80718)

X_train=np.array(x_train)
X_test=np.array(x_test)
Y_train=np.array(y_train)
Y_test=np.array(y_test)
