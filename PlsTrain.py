import pandas as pd
import scipy.sparse as sp
import joblib
from implicit.als import AlternatingLeastSquares


class ALStrain:
    def __init__(self, datarating):
        #clone và ép kiểu dữ liệu -> này để tránh lỗi dữ liệu
        datarating = datarating.copy()
        datarating['userId'] = datarating['userId'].astype(int)
        datarating['movieId'] = datarating['movieId'].astype(int)
        datarating['rating'] = datarating['rating'].astype(float)

        self.datarating = datarating[['userId', 'movieId', 'rating']] #đây sẽ là data matrix

        #Biến khởi tạo 
        self.model = None
        self.usermap = {}         
        self.itemmap = {}        
        self.user = {}            
        self.item = {}            
        self.useritemmx = None    
        self.alpha = None

    #Tạo ma trận để train nè 
    def buildMx(self, alpha=40):
        self.alpha = alpha

        # Ánh xạ 2 chiều userId và movieId ->ánh xạ đúng chiều để không bị ngược :'( 
        self.usermap = {u: i for i, u in enumerate(self.datarating['userId'].unique())}
        self.itemmap = {m: i for i, m in enumerate(self.datarating['movieId'].unique())}
        self.user = {i: u for u, i in self.usermap.items()}
        self.item = {i: m for m, i in self.itemmap.items()}

        # Lấy index theo map
        rows = self.datarating['userId'].map(self.usermap).astype(int)
        cols = self.datarating['movieId'].map(self.itemmap).astype(int)
        values = (1.0 + alpha * self.datarating['rating']).astype(float)

        # Đây sẽ tạo ma trận dạng user-item
        self.useritemmx = sp.coo_matrix(
            (values, (rows, cols)),
            shape=(len(self.usermap), len(self.itemmap))
        ).tocsr()
#khúc này check data đã đủ chưa, có mất mát hay hư gì không bằng việc đếm :(( có này để check với khi recommend
        print("TRAIN DEBUG để check")
        print("useritemmx:", self.useritemmx.shape)
        print("users:", len(self.usermap), "items:", len(self.itemmap))

    # Bắt đầu train mô hình ALS
    def trainALS(self, factors=50, iterations=15, regularization=0.1, alpha=40):
        if self.useritemmx is None:
            self.buildMx(alpha=alpha)

        model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            num_threads=0
        )

        # TRAIN với ma trận user × item
        model.fit(self.useritemmx)
        self.model = model
#Sau khi train xong xuất báo hiệu và số lượng thành phần để check có mất mát gì ko
        print("TRAINED")
        print("model.user_factors:", model.user_factors.shape)
        print("model.item_factors:", model.item_factors.shape)
        return model
    
    # Lưu file mô hình 
    def saveModel(self, path="als_model.pkl"):
        joblib.dump({
            "model": self.model,
            "user_mapping": self.usermap,
            "item_mapping": self.itemmap,
            "user_inv_mapping": self.user,
            "item_inv_mapping": self.item,
            "alpha": self.alpha
        }, path, compress=3)
        print(f"Saved to {path}")

#phần chạy train
if __name__ == "__main__":
    ratings = pd.read_csv("ratings.csv")
    trainer = ALStrain(ratings)
    print("Training ALS...") #đang train
    trainer.trainALS(factors=50, iterations=10, alpha=40)
    trainer.saveModel("als_model.pkl")
    print("Done") #báo xong
