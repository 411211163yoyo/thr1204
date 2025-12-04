import joblib

# 載入模型
rf_model = joblib.load("rf_model.pkl")       # RandomForest
xgb_model = joblib.load("xgb_model.pkl")     # 如果你有存 XGBoost

print("模型載入成功！")

# ======== 測試預測（你需要準備 X_test）========
# 這裡示範假設你有 test 資料
# X_test = ...

# 如果你還沒有 X_test，就先只測試模型屬性：
print("RandomForest 模型參數：")
print(rf_model.get_params())

# 如果 XGBoost 有載入：
try:
    print("\nXGBoost 模型參數：")
    print(xgb_model.get_params())
except:
    pass

print("程式執行完成！")
