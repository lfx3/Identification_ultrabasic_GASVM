# -*- coding: utf-8 -*-

# 代码 7-1

import pandas as pd
import joblib


#
print("Reading prediction data")
qinling_test = pd.read_csv('mag_radio_test.csv')
test_data = qinling_test.loc[:,["1","2","3","5"]]
num = qinling_test.iloc[:,10]


print("Loading Model")
model = joblib.load("GASVM_magnetic.pkl")

print("Predicting")
cifang_cla_pred = model.predict(test_data)

print("Outputing CSV")
dff = pd.DataFrame({"Index":num,"Predict_class":cifang_cla_pred[:]})
dff.to_csv("Magnetic_GASVM_result.csv",index=False)