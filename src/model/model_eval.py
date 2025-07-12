from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

class evaluate:
    def eval(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")
        print("MSE:",mse)

        return {"mse": mse, "rmse": rmse, "r2": r2}
