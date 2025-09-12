import os

import numpy as np
import scipy.io

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
from time import time
from tqdm import tqdm

class SpectralCNN1D(nn.Module):
    def __init__(self, input_length=261, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x : [batch, 261] -> [batch, 1, 261]
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # [N, 261]
        self.y = torch.tensor(y, dtype=torch.long)     # [N]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main(kind):

    # Training Data
    folder = r'C:\Users\Usuario\Documents\GitHub\Hypertool\identification\data'

    if kind == "3_inks_classes":
        data_filename= r'XY_train_3classes.mat'
        data = scipy.io.loadmat(os.path.join(folder,data_filename))
        X_train = data['X'] # spectra
        y_train = data['Y'].ravel() #labels

    elif "background" in kind:
        import h5py

        if "10pct" in kind :
            data_filename = r'XY_train_background_10pct.h5'

        else :
            data_filename = r'XY_train_background.mat'

        filename = os.path.join(folder, data_filename)
        data = h5py.File(filename, 'r')
        print(data.keys())
        X_train = np.array(data['X']).T
        y_train = np.array(data['Y'][0])


    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    print(f'[TRAIN] DATA LOADED')

    # <editor-fold desc="SVM">

    svm_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel='rbf',       # Gaussian kernel
            C=10,               # Box constraint
            gamma='scale',      # Automatic kernel scale
            decision_function_shape='ovo'
        )
    )
    start=time()
    print(f'[TRAIN] SVM fitting.....')
    svm_model.fit(X_train, y_train)
    modelname="model_svm_"+kind+".joblib"
    joblib.dump(svm_model, os.path.join(folder,modelname))
    stop=time()
    print(f'[TRAIN] SVM  in {stop - start:.2f} seconds')
    # </editor-fold>

    #<editor-fold desc="KNN">
    # KNN
    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=1,
            metric='cosine',
            weights='uniform'
        )
    )
    start=time()
    knn_model.fit(X_train, y_train)
    modelname="model_knn_"+kind+".joblib"
    joblib.dump(knn_model, os.path.join(folder,modelname))
    stop=time()
    print(f'[TRAIN] KNN  in {stop - start:.2f} seconds')
    # </editor-fold>

    # <editor-fold desc="LDA">
    start=time()
    lda_model = LinearDiscriminantAnalysis(solver='svd')
    lda_model.fit(X_train, y_train)
    modelname="model_lda_"+kind+".joblib"
    joblib.dump(lda_model, os.path.join(folder,modelname))
    stop=time()
    print(f'[TRAIN] LDA  in {stop - start:.2f} seconds')
    # </editor-fold>

    # <editor-fold desc="Random Forest">
    n_trees_total = 30
    rf_model = RandomForestClassifier(
        n_estimators=0,         # Start with 0 tree
        max_features=None,       # Use all predictors
        max_leaf_nodes=751266,   # Max number of splits
        bootstrap=True,
        warm_start=True,           # Allow incremental training
        n_jobs=-1,
    )

    start=time()
    for i in tqdm(range(1, n_trees_total + 1), desc="Training Random Forest"):
        rf_model.set_params(n_estimators=i)  # Increment tree count
        rf_model.fit(X_train, y_train)       # Train 1 extra tree
    modelname="model_rdf_"+kind+".joblib"
    joblib.dump(rf_model, os.path.join(folder,modelname))
    stop=time()
    print(f'[TRAIN] RDF  in {stop - start:.2f} seconds')
    # </editor-fold>

    # # <editor-fold desc="DeepLabV3 (PyTorch)">
    # input_length = 261  # 111 VNIR + 150 SWIR
    # num_classes = len(np.unique(y_train))  # à ajuster si besoin
    # learning_rate = 0.0001
    # epochs = 25
    # batch_size = 64
    #
    # # PyTorch dataset class
    # train_dataset = SpectralDataset(X_train, y_train)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    #
    # # ==== 4. Initialisation ====
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SpectralCNN1D(input_length, num_classes).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #
    # # ==== 5. Entraînement ====
    # start = time()
    #
    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0.0
    #     train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    #
    #     for inputs, labels in train_bar:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #         train_bar.set_postfix(loss=running_loss / (len(train_bar) or 1))
    #
    # # Sauvegarde du modèle
    # modelname="CNN1D_"+kind+".pth"
    # torch.save(model.state_dict(), os.path.join(folder, "CNN1D.pth"))
    #
    # stop = time()
    # print(f'[TRAIN] DPL in {stop - start:.2f} seconds')
    # # </editor-fold>

    print("All models have been trained and saved.")

if __name__=='__main__':
    kind="background" # "3_inks_classes" or "background" or "background_10pct"
    main(kind)