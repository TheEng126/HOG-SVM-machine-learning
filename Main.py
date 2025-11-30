import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

#load du lieu
def load_data_by_person(dataset_path, image_size=(64, 64),
                        train_ids=None, test_ids=None):
    X_train, y_train = [], []
    X_test, y_test = [], []
    label_map = {}
    label_id = 0

    for person_folder in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        person_index = int(person_folder)

        for gesture_folder in sorted(os.listdir(person_path)):
            gesture_path = os.path.join(person_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            if gesture_folder not in label_map:
                label_map[gesture_folder] = label_id
                label_id += 1

            for image_name in os.listdir(gesture_path):
                if not (image_name.endswith('.png') or
                        image_name.endswith('.jpg') or
                        image_name.endswith('.jpeg')):
                    continue

                image_path = os.path.join(gesture_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, image_size)

                if person_index in train_ids:
                    X_train.append(img)
                    y_train.append(label_map[gesture_folder])
                elif person_index in test_ids:
                    X_test.append(img)
                    y_test.append(label_map[gesture_folder])

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test), label_map)

#trich chon dac trung HOG
def trich_chon_hog(image, cell_size=8, block_size=2, bins=9):
    #B1
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = np.float32(image) / 255.0
    h, w = image.shape

    #B2
    Dx = np.array([[-1, 0, 1]], dtype=np.float32)
    Dy = np.array([[-1], [0], [1]], dtype=np.float32)

    Ix = cv2.filter2D(image, cv2.CV_32F, Dx)
    Iy = cv2.filter2D(image, cv2.CV_32F, Dy)

    magnitude = np.sqrt(Ix**2 + Iy**2)
    angle = np.arctan2(Iy, Ix) * 180 / np.pi
    angle = angle % 180  # [0, 180)

    #B3
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size

    orientation_histogram = np.zeros((n_cells_y, n_cells_x, bins))
    bin_width = 180.0 / bins

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            y_start, y_end = i * cell_size, (i + 1) * cell_size
            x_start, x_end = j * cell_size, (j + 1) * cell_size

            cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
            cell_angle = angle[y_start:y_end, x_start:x_end]

            hist = np.zeros(bins)

            for yy in range(cell_size):
                for xx in range(cell_size):
                    mag = cell_magnitude[yy, xx]
                    ang = cell_angle[yy, xx]

                    bin_index = ang / bin_width
                    lower_bin = int(np.floor(bin_index)) % bins
                    upper_bin = (lower_bin + 1) % bins

                    bin_center_lower = lower_bin * bin_width
                    ratio = (ang - bin_center_lower) / bin_width

                    hist[lower_bin] += mag * (1 - ratio)
                    hist[upper_bin] += mag * ratio

            orientation_histogram[i, j] = hist

    #B4
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1

    hog_vector = []

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            block = orientation_histogram[y:y+block_size, x:x+block_size]
            block_vector = block.ravel()

            norm = np.sqrt(np.sum(block_vector**2) + 1e-5)
            block_normalized = block_vector / norm

            hog_vector.extend(block_normalized)

    #B5
    hog_feature_vector = np.array(hog_vector)
    return hog_feature_vector

#trich xuat HOG cho cac anh
def xuat_hog(images, cell_size=8, block_size=2, bins=9):
    hog_features = []
    for img in images:
        hog_feat = trich_chon_hog(img, cell_size, block_size, bins)
        hog_features.append(hog_feat)
    return np.array(hog_features)

#SVM tuyen tinh theo one-vs-rest
class SVM:
    def __init__(self, C=1.0, lr=1e-3, n_epochs=200, verbose=True):
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.W = None
        self.b = None
        self.classes_ = None

    def fit(self, X, y):
        """
        X: (N, d)  - đặc trưng (ví dụ PCA(HOG))
        y: (N,)    - nhãn 0..K-1
        """
        N, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        self.W = np.zeros((K, d), dtype=np.float32)
        self.b = np.zeros(K, dtype=np.float32)

        for epoch in range(self.n_epochs):
            for idx, cls in enumerate(self.classes_):
                y_bin = np.where(y == cls, 1.0, -1.0)

                w = self.W[idx]
                b = self.b[idx]

                scores = X @ w + b
                margins = y_bin * scores

                mask = margins < 1

                if np.any(mask):
                    grad_w = w - self.C * np.sum(
                        y_bin[mask][:, None] * X[mask], axis=0
                    )
                    grad_b = - self.C * np.sum(y_bin[mask])
                else:
                    grad_w = w
                    grad_b = 0.0

                w -= self.lr * grad_w
                b -= self.lr * grad_b

                self.W[idx] = w
                self.b[idx] = b

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"[Manual SVM] Epoch {epoch+1}/{self.n_epochs} done.")

        return self

    def decision_function(self, X):
        return X @ self.W.T + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]
    
#truc quan hoa qua trinh thuc hien HOG
def hienthi_hog(image_path, cell_size=8, block_size=2, bins=9): 
    """
    Hiển thị từng bước HOG cho 1 ảnh:
    B1: Ảnh gốc -> resize 64x64 (grayscale)
    B2: Ix, Iy, magnitude của gradient
    B3: Histogram 9 bin của 1 cell bất kỳ (cell [0,0])
    B4: Ảnh HOG (hướng gradient vẽ trên ảnh 64x64)
    B5: Vector HOG tổng của ảnh
    """
    # Đọc ảnh
    img = cv2.imread(image_path)

    # ---------------- BƯỚC 1: TIỀN XỬ LÝ (gốc -> 64x64) ----------------
    if img is None:
        print("Không đọc được ảnh:", image_path)
        return

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray_64 = cv2.resize(gray, (64, 64))        # resize về 64x64
    gray_64_float = np.float32(gray_64) / 255.0 # chuẩn hóa [0,1] để tính gradient

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.title("Ảnh gốc")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(1,2,2)
    plt.title("B1: Gray 64x64")
    plt.axis("off")
    plt.imshow(gray_64, cmap="gray")
    plt.tight_layout()
    plt.show()

    # ---------------- BƯỚC 2: TÍNH GRADIENT ----------------
    Dx = np.array([[-1, 0, 1]], dtype=np.float32)
    Dy = np.array([[-1],
                   [0],
                   [1]], dtype=np.float32)

    Ix = cv2.filter2D(gray_64_float, cv2.CV_32F, Dx)
    Iy = cv2.filter2D(gray_64_float, cv2.CV_32F, Dy)
    magnitude = np.sqrt(Ix**2 + Iy**2)

    # Chuẩn hoá về [0,1] để hiển thị đẹp
    def norm01(x):
        x = np.abs(x)
        m = x.max()
        if m < 1e-6:
            return x
        return x / m

    Ix_show = norm01(Ix)
    Iy_show = norm01(Iy)
    mag_show = norm01(magnitude)

    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.title("B2: |Ix|")
    plt.axis("off")
    plt.imshow(Ix_show, cmap="gray")

    plt.subplot(1,3,2)
    plt.title("B2: |Iy|")
    plt.axis("off")
    plt.imshow(Iy_show, cmap="gray")

    plt.subplot(1,3,3)
    plt.title("B2: Magnitude")
    plt.axis("off")
    plt.imshow(mag_show, cmap="gray")

    plt.tight_layout()
    plt.show()

    # ---------------- BƯỚC 3: HISTOGRAM 9 BINS CỦA CÁC CELLS ----------------
    h, w = gray_64_float.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    angle = np.arctan2(Iy, Ix) * 180 / np.pi
    angle = angle % 180.0

    orientation_histogram = np.zeros((n_cells_y, n_cells_x, bins), dtype=np.float32)
    bin_width = 180.0 / bins

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            y_start = i * cell_size
            y_end   = (i + 1) * cell_size
            x_start = j * cell_size
            x_end   = (j + 1) * cell_size

            cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
            cell_angle     = angle[y_start:y_end, x_start:x_end]

            hist = np.zeros(bins, dtype=np.float32)

            for yy in range(cell_size):
                for xx in range(cell_size):
                    mag = cell_magnitude[yy, xx]
                    ang = cell_angle[yy, xx]

                    bin_index = ang / bin_width
                    lower_bin = int(np.floor(bin_index)) % bins
                    upper_bin = (lower_bin + 1) % bins

                    ratio = (ang - lower_bin * bin_width) / bin_width
                    hist[lower_bin] += mag * (1 - ratio)
                    hist[upper_bin] += mag * ratio

            orientation_histogram[i, j] = hist

    # Lấy histogram của 1 cell bất kỳ
    cell_hist = orientation_histogram[3, 6]

    plt.figure(figsize=(4,3))
    plt.title("B3: Histogram 9 bins của cell (0,0)")
    plt.bar(np.arange(bins), cell_hist)
    plt.xlabel("Bin (0..8)")
    plt.ylabel("Giá trị vote (magnitude)")
    plt.tight_layout()
    plt.show()

    # ---------------- BƯỚC 4: VẼ ẢNH HOG ĐÈ LÊN ẢNH 64x64 ----------------
    # ảnh nền là gray_64, chuyển sang 3 kênh để vẽ line trắng
    overlay = cv2.cvtColor(gray_64, cv2.COLOR_GRAY2BGR)
    overlay = np.float32(overlay) / 255.0

    max_hist = orientation_histogram.max()
    if max_hist < 1e-6:
        max_hist = 1.0

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            hist = orientation_histogram[i, j]

            # Tâm của cell (i, j)
            cy = i * cell_size + cell_size // 2
            cx = j * cell_size + cell_size // 2

            for b in range(bins):
                val = hist[b]
                if val <= 0:
                    continue

                # góc trung tâm của bin b
                angle_deg = (b + 0.5) * bin_width      # 0..180
                angle_rad = np.deg2rad(angle_deg)

                # độ dài đoạn thẳng tỉ lệ với giá trị bin
                length = (val / max_hist) * (cell_size / 2)

                dx = length * np.cos(angle_rad)
                dy = length * np.sin(angle_rad)

                x1 = int(cx - dx)
                y1 = int(cy - dy)
                x2 = int(cx + dx)
                y2 = int(cy + dy)

                # vẽ line màu trắng (1,1,1)
                cv2.line(overlay, (x1, y1), (x2, y2), (1.0, 1.0, 1.0), 1)

    plt.figure(figsize=(4,4))
    plt.title("B4: HOG directions on 64x64 image")
    plt.axis("off")
    plt.imshow(overlay)
    plt.tight_layout()
    plt.show()

    hog_vec = trich_chon_hog(gray_64, cell_size=cell_size, block_size=block_size, bins=bins)

    print("B5: Vector HOG tổng của ảnh")
    print("  Độ dài vector:", hog_vec.shape[0])

    plt.figure(figsize=(8,3))
    plt.title("B5: Vector HOG tổng (1D)")
    plt.plot(hog_vec)
    plt.xlabel("Chỉ số đặc trưng")
    plt.ylabel("Giá trị")
    plt.tight_layout()
    plt.show()

def main():
    data_dir = 'C:\\Users\\thean\\Downloads\\BAI_TAP_LON\\leapGestRecog'
    train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    test_ids = [9]

    #load dataset
    X_train_img, y_train, X_test_img, y_test, label_map = load_data_by_person(
        data_dir, train_ids=train_ids, test_ids=test_ids
    )
    print("Train samples:", len(X_train_img), " Test samples:", len(X_test_img))

    #trich xuat HOG
    X_train = xuat_hog(X_train_img)
    X_test = xuat_hog(X_test_img)

    #PCA (tuy chon)
    """pca = PCA(n_components=0.95, svd_solver='full')
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)
    print("Original dim:", X_train_raw.shape[1])
    sample_idx = 0
    vec_pca = X_train[sample_idx]"""

    #plt.figure(figsize=(12, 4))
    #plt.plot(vec_pca)
    #plt.xlabel("Chỉ số đặc trưng sau PCA")
    #plt.ylabel("Giá trị")
    #plt.title(f"Vector đặc trưng sau PCA (số chiều = {vec_pca.shape[0]})")
    #plt.tight_layout()
    #plt.show()
    #print("Số chiều vector HOG sau PCA:", vec_pca.shape[0])

    hienthi_hog("C:\\Users\\thean\\Downloads\\BAI_TAP_LON\\leapGestRecog\\01\\03_fist\\frame_01_03_0003.png")

    #train SVM
    model = SVM(C=1.0, lr=1e-3, n_epochs=200, verbose=True)
    model.fit(X_train, y_train)

    #hien thi ket qua
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    labels = list(label_map.keys())

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for i in range(10):
        idx = np.random.randint(0, len(X_test_img))
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test_img[idx], cmap='gray')
        true_label = [k for k, v in label_map.items()
                      if v == y_test[idx]][0]
        pred_label = [k for k, v in label_map.items()
                      if v == y_pred[idx]][0]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()