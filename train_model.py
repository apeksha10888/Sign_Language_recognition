import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib



X=[]
y=[]



for file in os.listdir("dataset"):

    if file.endswith(".npy"):

        label=file.replace(".npy","")


        data=np.load(
            "dataset/"+file
        )


        for sample in data:

            X.append(sample)
            y.append(label)



X=np.array(X)


model=RandomForestClassifier(
    n_estimators=100
)



model.fit(
    X,
    y
)



joblib.dump(
    model,
    "models/sign_model.pkl"
)



print("Training complete")
print("Samples:",len(X))
