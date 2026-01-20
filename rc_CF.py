import scipy.sparse as sp
import joblib
import os
from difflib import get_close_matches


class ALSRec:
    def __init__(self, datarating, datamovie=None, path="als_model.pkl", alpha=40): #khúc chạy thuật toán recommend nè
        datarating = datarating.copy()
        datarating['userId'] = datarating['userId'].astype(int)
        datarating['movieId'] = datarating['movieId'].astype(int)
        datarating['rating'] = datarating['rating'].astype(float)

        self.datarating = datarating[['userId', 'movieId', 'rating']]
        self.datamovie = None
        if datamovie is not None:
            self.datamovie = datamovie.copy()
            self.datamovie['movieId'] = self.datamovie['movieId'].astype(int)
            if 'title' in self.datamovie.columns:
                self.datamovie['title'] = self.datamovie['title'].astype(str)

        if not os.path.exists(path):
            raise FileNotFoundError(f" Không tìm thấy model tại {path}")
        obj = joblib.load(path)

        self.model = obj["model"]
        self.usermap = obj["user_mapping"]
        self.itemmap = obj["item_mapping"]
        self.user = obj["user_inv_mapping"]
        self.item = obj["item_inv_mapping"]
        self.alpha = obj.get("alpha", alpha)

        # Debug lại map do lỗi data khi mapping :((( // xét và đếm thành phần sao cho đã giống vs file data gốc chưa
        print("CHECk cái user_mapping size:", len(self.usermap)) 
        print("min user index:", min(self.usermap.values()))
        print("max user index:", max(self.usermap.values()))
        print("min userId in mapping:", min(self.user.values()))
        print("max userId in mapping:", max(self.user.values()))

        print("CHECK cái item_mapping size:", len(self.itemmap))
        print("min item index:", min(self.itemmap.values()))
        print("max item index:", max(self.itemmap.values()))
        print("min movieId in mapping:", min(self.item.values()))
        print("max movieId in mapping:", max(self.item.values()))

        print("CHECK cái ratings.csv size:", len(self.datarating))
        print("min userId in ratings:", self.datarating['userId'].min())
        print("max userId in ratings:", self.datarating['userId'].max())
        print("min movieId in ratings:", self.datarating['movieId'].min())
        print("max movieId in ratings:", self.datarating['movieId'].max())

        # Check dimensions
        mu = self.model.user_factors.shape[0]
        mi = self.model.item_factors.shape[0]
        lu = len(self.usermap)
        li = len(self.itemmap)

        if not (lu == mu and li == mi):
            raise RuntimeError( #báo error nếu mismatch
                f"Mapping-size mismatch.\n"
                f"mapping users={lu}, items={li}\n"
                f"model.user_factors={mu}, model.item_factors={mi}"
            )

        # Build matran
        self.buildMx(alpha=self.alpha)
        if self.useritemmx.shape[0] == self.model.user_factors.shape[0]:
            self.useritemmodel = self.useritemmx
        elif self.item_user_matrix.shape[0] == self.model.user_factors.shape[0]:
            self.useritemmodel = self.item_user_matrix
        else:
            raise RuntimeError(
                f"Neither matrix has rows == model.user_factors.\n"
                f"useritemmx: {self.useritemmx.shape}, "
                f"item_user_matrix: {self.item_user_matrix.shape}, "
                f"model.user_factors: {self.model.user_factors.shape[0]}"
            )
# Khúc này lại debug lại sau khi build để so sánh lại với trước khi build để coi đã chiếu đúng chưa
        print("DEBUG MATRIX (để check)")
        print("useritemmx shape:", self.useritemmx.shape)
        print("model expects users:", self.model.user_factors.shape[0])
        print("model expects items:", self.model.item_factors.shape[0])
#Hàm build ma trận
    def buildMx(self, alpha=40):
        num_users = len(self.usermap)
        num_items = len(self.itemmap)

        rows = self.datarating['userId'].map(self.usermap)
        cols = self.datarating['movieId'].map(self.itemmap)

        if rows.isna().any() or cols.isna().any():
            bad_u = self.datarating.loc[rows.isna(), 'userId'].unique()[:5]
            bad_i = self.datarating.loc[cols.isna(), 'movieId'].unique()[:5]
            raise RuntimeError(
                f"Có ID không map được. Ví dụ userId={bad_u}, movieId={bad_i}. "
                "Hãy check lại xem tập train với tập data khớp chưa."
            )

        rows = rows.astype(int).to_numpy()
        cols = cols.astype(int).to_numpy()
        values = (1.0 + alpha * self.datarating['rating']).astype(float).to_numpy()

        self.useritemmx = sp.coo_matrix(
            (values, (rows, cols)),
            shape=(num_users, num_items)
        ).tocsr()
        self.item_user_matrix = self.useritemmx.T.tocsr()

    def recommenduser(self, user_id, top_k=10):
        if user_id not in self.usermap:
            raise ValueError(f"User {user_id} không tồn tại trong training data")

        user_idx = self.usermap[user_id]
        print(f"Check lại recommenduser: user_id={user_id}, user_idx={user_idx}") #Lại check xem đã đúng chưa :'((((((

        user_vector = self.useritemmx[user_idx]

        rec = self.model.recommend(
            userid=user_idx,
            user_items=user_vector,
            N=top_k,
            filter_already_liked_items=True
        )

        result = []
        if isinstance(rec, tuple):
            item_ids, scores = rec
            for iid, score in zip(item_ids, scores):
                if iid in self.item:
                    mid = int(self.item[iid])
                    title = self.datamovie.loc[self.datamovie['movieId'] == mid, 'title'].values[0]
                    result.append((mid, str(title), float(score)))
        else:
            for iid, score in rec:
                if iid in self.item:
                    mid = int(self.item[iid])
                    title = self.datamovie.loc[self.datamovie['movieId'] == mid, 'title'].values[0]
                    result.append((mid, str(title), float(score)))

        return result


#Phần của item-based -> xài similiar cx như xài fuzzy để nhận được tên gần đúng của phim
    def simitem(self, movie_name, top_k=10):
        if self.datamovie is None:
            raise RuntimeError("Cần datamovie để tìm phim")
# phần kiểm tra tên
        match = self.datamovie[self.datamovie['title'].str.contains(movie_name, case=False, regex=False)]
        if match.empty:
            all_titles = self.datamovie['title'].tolist()
            close = get_close_matches(movie_name, all_titles, n=1, cutoff=0.6)
            if close:
                match = self.datamovie[self.datamovie['title'] == close[0]]
        if match.empty:
            raise ValueError(f"Không tìm thấy phim gần đúng với: {movie_name}")
        movie_id = int(match.iloc[0]['movieId'])
        if movie_id not in self.itemmap:
            raise ValueError(f"MovieId {movie_id} không có trong training data")
        movie_idx = self.itemmap[movie_id]
        sim = self.model.similar_items(movie_idx, N=top_k + 5)
        if isinstance(sim, tuple):
            item_ids, scores = sim
            pair = list(zip(item_ids, scores))
        else:
            pair = sim
        result = []
        for iid, score in pair:
            if iid not in self.item:
                continue
            mid = int(self.item[iid])
            title = self.datamovie.loc[self.datamovie['movieId'] == mid, 'title'].values[0]
            result.append((mid, str(title), float(score)))
            if len(result) >= top_k:
                break
        return result
