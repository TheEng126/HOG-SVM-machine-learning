import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.keras import layers

from Main import (
    load_data_by_person,
    xuat_hog,
    SVM,
)

#thiet lap CNN co ban
def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((input_shape[0], input_shape[1], 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    data_dir = r"C:\\Users\\thean\\Downloads\\BAI_TAP_LON\\leapGestRecog"
    train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    test_ids = [9]

    X_train_img, y_train, X_test_img, y_test, label_map = load_data_by_person(
        data_dir, train_ids=train_ids, test_ids=test_ids
    )
    print("Train samples:", len(X_train_img), "Test samples:", len(X_test_img))

    #HOG cho SVM va KNN
    X_train_hog = xuat_hog(X_train_img)
    X_test_hog = xuat_hog(X_test_img)

    #du lieu anh goc cho CNN
    X_train_cnn = np.array(X_train_img) / 255.0
    X_test_cnn = np.array(X_test_img) / 255.0
    
    labels = [gesture for gesture, idx in sorted(label_map.items(),
                                                 key=lambda x: x[1])]
    num_classes = len(labels)

    #SVM
    print("\n=== Training Manual Linear SVM ===")
    manual_svm = SVM(C=1.0, lr=1e-3, n_epochs=200, verbose=True)
    manual_svm.fit(X_train_hog, y_train)
    y_pred_svm = manual_svm.predict(X_test_hog)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"[Manual Linear SVM] Accuracy: {acc_svm*100:.2f}%")

    #CNN
    print("\n=== Training CNN ===")
    img_shape = X_train_cnn[0].shape
    cnn_model = create_cnn_model(img_shape, num_classes)
    cnn_model.fit(X_train_cnn, y_train, 
                  epochs=20, 
                  batch_size=32,
                  validation_split=0.2,
                  verbose=1)
    
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    print(f"[CNN] Accuracy: {acc_cnn*100:.2f}%")

    #KNN
    print("\n=== Training KNN ===")
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_train_hog, y_train)
    y_pred_knn = knn.predict(X_test_hog)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    print(f"[KNN (k=5)] Accuracy: {acc_knn*100:.2f}%")

    #so sanh
    print("\n=== COMPARISON ===")
    print(f"Manual Linear SVM: {acc_svm*100:.2f}%")
    print(f"CNN:               {acc_cnn*100:.2f}%")
    print(f"KNN (k=5):         {acc_knn*100:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    #ma tran nham lan SVM
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_svm,
                                   display_labels=labels)
    disp1.plot(cmap="Blues", ax=axes[0], xticks_rotation=45)
    axes[0].set_title(f"Manual Linear SVM\nAccuracy: {acc_svm*100:.2f}%")

    #ma tran nham lan CNN
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_cnn,
                                   display_labels=labels)
    disp2.plot(cmap="Greens", ax=axes[1], xticks_rotation=45)
    axes[1].set_title(f"CNN\nAccuracy: {acc_cnn*100:.2f}%")

    #ma tran nham lan KNN
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_knn,
                                   display_labels=labels)
    disp3.plot(cmap="Oranges", ax=axes[2], xticks_rotation=45)
    axes[2].set_title(f"KNN (k=5)\nAccuracy: {acc_knn*100:.2f}%")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()