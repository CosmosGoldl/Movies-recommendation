from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import os
from rc_CF import ALSRec

app = Flask(__name__)

OMDB_API_KEY = "e54bd3ba"  # API key OMDB

# Load dữ liệu file csv vào
drating = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int32"
}
dmovie = {
    "movieId": "int32",
    "title": "string",
    "genres": "string"
}
dlink = {
    "movieId": "int32",
    "imdbId": "Int64",
    "tmdbId": "Int64"
}
#read dữ liệu bằng thư viện panda
rating = pd.read_csv("ratings.csv", dtype=drating)
movie = pd.read_csv("movies.csv", dtype=dmovie)
link = pd.read_csv("links.csv", dtype=dlink)

#Chạy ALS
modpath = "als_model.pkl"
if not os.path.exists(modpath):
    raise FileNotFoundError("Chưa có model đâu! Hãy chạy file PlsTrain trước.") #không có model sao chạy đc
cf = ALSRec(rating, movie, path=modpath, alpha=40)
print("ALS Model đã load thành công")

# INFO
def movieinfo(rec):  # lấy info phim về bằng omdb do csv thiếu thông tin phim
    info = []
    for movieid, title, score in rec:  
        # Lấy imdbId từ link.csv
        rowlink = link.loc[link['movieId'] == movieid]
        imdbid = None
        if not rowlink.empty and pd.notna(rowlink['imdbId'].values[0]):
            imdbid = f"tt{int(rowlink['imdbId'].values[0]):07d}"

        # Gọi OMDB API
        omdb = {}
        if imdbid:
            url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&i={imdbid}&plot=short"
            try:
                omdb = requests.get(url, timeout=5).json()
            except:
                omdb = {}

        info.append({
            "movieId": int(movieid),
            "title": title,
            "score": round(float(score), 3),
            "imdbId": imdbid,
            "poster": omdb.get("Poster", ""),
            "year": omdb.get("Year", ""),
            "plot": omdb.get("Plot", ""),
            "actors": omdb.get("Actors", ""),
            "imdbRating": omdb.get("imdbRating", "N/A")
        })
    return info

def cfuser(userid): #rec user
    rec = cf.recommenduser(userid, top_k=10) #mình lấy 10 bộ
    return movieinfo(rec)

def cfitem(moviename): #rec item
    rec = cf.simitem(moviename, top_k=10) #mình lấy 10 bộ
    return movieinfo(rec)

#UI 
@app.route("/")  # Trang Home
def home():
    return render_template("index.html")

@app.route("/recommend/<int:userid>") #phần user 
def rcUI(userid):
    try:
        info = cfuser(userid)
        return render_template("recommend.html", mode="user", user_id=userid, movies=info)
    except ValueError as e:
        return render_template("error.html", message=str(e))


@app.route("/similar/<moviename>")#phần item
def simUI(moviename):
    try:
        info = cfitem(moviename)
        return render_template("recommend.html", mode="item", query=moviename, movies=info)
    except ValueError as e:
        return render_template("error.html", message=str(e))

#API 
@app.route("/api/rc/<int:userid>")
def rcapi(userid):
    try:
        info = cfuser(userid)
        return jsonify(info)
    except ValueError as e:
        return jsonify({"error": str(e)}), 777

@app.route("/api/sim/<path:moviename>")
def simapi(moviename):
    print("Flask nhận request với moviename =", moviename) #cái này để debug xem đã nhận phim với tên gì (check cái name)
    try:
        info = cfitem(moviename)
        return jsonify(info)
    except ValueError as e:
        print("Lỗi trong simitem:", e) 
        return jsonify({"error": str(e)}), 777
    
#Main   
if __name__ == "__main__":
    print("Các route đang có trong Flask:") #debug xem có nhận sai dữ liệu không khi nhập vào box :') =>quan trọng
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True)
