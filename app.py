import sys
import os

# Thêm đường dẫn hiện tại vào hệ thống TRƯỚC khi import các module khác
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
from rc_CF import ALSRec
from contentbased import ContentBasedRec
from hybrid_recommender import HybridRecommender
from difflib import get_close_matches
import threading

app = Flask(__name__)

# --- CONFIGURATION ---
OMDB_API_KEY = "a4e75082"

# --- ANCHOR MOVIES CONFIGURATION ---
ANCHOR_PRIMARY_GENRE = {
    "Matrix, The (1999)": "Sci-Fi",
    "Gladiator (2000)": "Action",
    "Memento (2000)": "Thriller",
    "Lord of the Rings: The Fellowship of the Ring, The (2001)": "Adventure",
    "Shrek (2001)": "Animation",
    "Beautiful Mind, A (2001)": "Drama",
    "Spider-Man (2002)": "Action",
    "Lord of the Rings: The Two Towers, The (2002)": "Adventure",
    "Finding Nemo (2003)": "Animation",
    "Lord of the Rings: The Return of the King, The (2003)": "Adventure",
    "Dark Knight, The (2008)": "Action",
    "Iron Man (2008)": "Action",
    "WALL·E (2008)": "Animation",
    "Slumdog Millionaire (2008)": "Drama",
    "Avatar (2009)": "Sci-Fi",
    "Inception (2010)": "Sci-Fi",
    "Toy Story 3 (2010)": "Animation",
    "Avengers, The (2012)": "Action",
    "Django Unchained (2012)": "Drama",
    "Interstellar (2014)": "Sci-Fi",
    "Guardians of the Galaxy (2014)": "Sci-Fi",
    "Mad Max: Fury Road (2015)": "Action",
    "Inside Out (2015)": "Animation",
    "La La Land (2016)": "Comedy",
    "Parasite (2019)": "Drama",
    "Empire Strikes Back, The (1980)": "Sci-Fi",
    "Raiders of the Lost Ark (1981)": "Adventure",
    "E.T. the Extra-Terrestrial (1982)": "Sci-Fi",
    "Blade Runner (1982)": "Sci-Fi",
    "Back to the Future (1985)": "Sci-Fi",
    "Princess Bride, The (1987)": "Adventure",
    "Indiana Jones and the Last Crusade (1989)": "Adventure",
    "Terminator 2: Judgment Day (1991)": "Action",
    "Beauty and the Beast (1991)": "Animation",
    "Jurassic Park (1993)": "Adventure",
    "Shawshank Redemption, The (1994)": "Drama",
    "Pulp Fiction (1994)": "Drama",
    "Forrest Gump (1994)": "Drama",
    "Toy Story (1995)": "Animation",
    "Titanic (1997)": "Drama"
}

ANCHOR_TARGETS = {
    "Sci-Fi": 3,
    "Action": 3,
    "Adventure": 3,
    "Drama": 3,
    "Animation": 2,
    "Comedy": 2
}

# --- THREAD-SAFE CACHE FOR OMDB API ---
omdb_cache = {}  # Cache OMDB results: {imdbId: omdb_response}
omdb_cache_lock = threading.Lock()  # Thread-safe lock for cache

# --- SHARED UTILITIES ---
class MovieSearchUtils:
    """Tiện ích dùng chung để xử lý tên phim cho các hệ thống gợi ý khác nhau"""
    
    @staticmethod
    def normalize_and_find_movie(movie_title, content_based_rec, als_rec, datamovie, prefer_content_based=True):
        if not movie_title or not isinstance(movie_title, str):
            return False, None
            
        movie_title = movie_title.strip()
        
        # Cách 1: Thử tìm kiếm nâng cao bằng Content-Based
        if prefer_content_based and content_based_rec:
            try:
                candidates = content_based_rec.find_movie_candidates(movie_title)
                if candidates:
                    best_match = candidates[0]  # (index, title, score, year)
                    movie_row = content_based_rec.movies.iloc[best_match[0]]
                    return True, {
                        'movieId': movie_row['movieId'],
                        'title': movie_row['title'],
                        'score': best_match[2],
                        'search_method': 'content_based_fuzzy'
                    }
            except Exception as e:
                print(f"Content-based search failed: {e}")
        
        # Cách 2: Fallback tìm kiếm theo Collaborative Filtering (regex & fuzzy match)
        if als_rec and datamovie is not None:
            try:
                match = datamovie[datamovie['title'].str.contains(movie_title, case=False, regex=False)]
                if match.empty:
                    all_titles = datamovie['title'].tolist()
                    close = get_close_matches(movie_title, all_titles, n=1, cutoff=0.6)
                    if close:
                        match = datamovie[datamovie['title'] == close[0]]
                
                if not match.empty:
                    movie_row = match.iloc[0]
                    return True, {
                        'movieId': movie_row['movieId'],
                        'title': movie_row['title'], 
                        'score': 0.8,
                        'search_method': 'collaborative_filtering'
                    }
            except Exception as e:
                print(f"Collaborative filtering search failed: {e}")
        
        return False, None

# --- 1. LOAD DATA ---
drating = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"}
dmovie = {"movieId": "int32", "title": "string", "genres": "string"}
dlink = {"movieId": "int32", "imdbId": "Int64", "tmdbId": "Int64"}

rating = pd.read_csv("ratings.csv", dtype=drating)
movie = pd.read_csv("movies.csv", dtype=dmovie)
link = pd.read_csv("links.csv", dtype=dlink)

# --- 2. LOAD MODELS ---
# ALS Model (Collaborative Filtering)
modpath = "als_model.pkl"
if not os.path.exists(modpath):
    raise FileNotFoundError("Chưa có model ALS! Hãy chạy file PlsTrain trước.")
cf = ALSRec(rating, movie, path=modpath, alpha=40)
print("ALS Model đã load thành công")

# Content-Based Model
cb = ContentBasedRec(movie)
print("Content-Based Model đã load thành công")

# Hybrid Recommender
hybrid = HybridRecommender(cf, rating, movie)
print("Hybrid Recommender đã khởi tạo thành công")

# --- CACHE OPTIMIZATION ---
# Cache avg_ratings at startup (only calculate once)
print("Đang tính toán avg_ratings...")
avg_ratings_cache = rating.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
avg_ratings_cache.columns = ["movieId", "avg_rating", "rating_count"]
print(f"✅ Cached avg_ratings for {len(avg_ratings_cache)} movies")

# Clear old OMDB cache on startup (remove stale error responses)
print("Clearing old OMDB cache...")
omdb_cache.clear()
print("✅ OMDB cache cleared - fresh start")

# --- 3. HELPER FUNCTIONS ---
# Track cache stats
cache_stats_data = {'hits': 0, 'misses': 0}
cache_stats_lock = threading.Lock()

# Sentinel for cache fetch tracking (must be module-level)
_FETCHING = object()  # Single sentinel object for all threads

def movieinfo(rec):
    """Lấy thông tin chi tiết phim từ OMDB API với cache"""
    info = []
    
    for movieid, title, score in rec:
        rowlink = link.loc[link['movieId'] == movieid]
        imdbid = None
        if not rowlink.empty and pd.notna(rowlink['imdbId'].values[0]):
            imdbid = f"tt{int(rowlink['imdbId'].values[0]):07d}"

        omdb = {}
        if imdbid:
            # Check cache với global sentinel
            should_fetch = False
            with omdb_cache_lock:
                if imdbid in omdb_cache:
                    cached_value = omdb_cache[imdbid]
                    if cached_value is not _FETCHING:
                        omdb = cached_value
                        # Track cache hit
                        with cache_stats_lock:
                            cache_stats_data['hits'] += 1
                    else:
                        # Đang fetch, fallback
                        should_fetch = False
                        omdb = {}
                else:
                    # Mark là đang fetch
                    omdb_cache[imdbid] = _FETCHING
                    should_fetch = True
                    # Track cache miss
                    with cache_stats_lock:
                        cache_stats_data['misses'] += 1
            
            # Fetch outside lock
            if should_fetch:
                url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&i={imdbid}&plot=short"
                try:
                    resp = requests.get(url, timeout=5)
                    omdb = resp.json()
                    print(f"OMDB fetch: {imdbid} -> Response={omdb.get('Response', 'N/A')}, Error={omdb.get('Error', 'None')}")
                    # Lưu kết quả vào cache (thay thế _FETCHING)
                    if omdb.get('Response') == 'True':
                        with omdb_cache_lock:
                            omdb_cache[imdbid] = omdb
                    else:
                        # Don't cache errors - let next request retry
                        print(f"⚠️ OMDB failed for {imdbid}: {omdb.get('Error', 'Unknown error')} - NOT CACHING")
                        with omdb_cache_lock:
                            omdb_cache.pop(imdbid, None)
                        omdb = {}  # Return empty dict instead of error response
                except Exception as e:
                    print(f"❌ OMDB exception for {imdbid}: {e}")
                    omdb = {}
                    # Remove sentinel on error
                    with omdb_cache_lock:
                        omdb_cache.pop(imdbid, None)

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

def cfuser(userid):
    try:
        rec = cf.recommenduser(userid, top_k=10)
        return movieinfo(rec)
    except Exception as e:
        print(f"Error in cfuser: {e}")
        raise

def cfitem(moviename):
    found, movie_data = MovieSearchUtils.normalize_and_find_movie(moviename, cb, cf, movie)
    if not found:
        raise ValueError(f"Không tìm thấy phim '{moviename}'.")
    
    resolved_title = movie_data['title']
    rec = cf.simitem(resolved_title, top_k=10)
    return movieinfo(rec)

def adaptive_hybrid_item(moviename, cf_threshold=0.6):
    """
    Adaptive Hybrid recommendation cho similar movies
    Tự động điều chỉnh giữa CF và Content-based dựa trên độ tin cậy của CF
    """
    found, movie_data = MovieSearchUtils.normalize_and_find_movie(moviename, cb, cf, movie)
    if not found:
        raise ValueError(f"Không tìm thấy phim '{moviename}'.")
    
    resolved_title = movie_data['title']
    
    rec = hybrid.adaptive_hybrid_simitem(
        resolved_title, 
        content_based_rec=cb, 
        top_k=10,
        cf_threshold=cf_threshold
    )
    
    # Convert kết quả để phù hợp với movieinfo format
    # rec format có thể là: (movie_id, title, score, method, cf_confidence) hoặc 
    # (movie_id, title, score, method, cf_confidence, cf_weight, content_weight) cho result đầu tiên
    if rec:
        formatted_rec = []
        for i, result in enumerate(rec):
            # Lấy 3 elements đầu tiên cho movieinfo: (movie_id, title, score)
            formatted_rec.append((result[0], result[1], result[2]))
    else:
        formatted_rec = []
    
    # Thêm thông tin về method và confidence vào response
    info = movieinfo(formatted_rec)
    
    # Thêm metadata về hybrid strategy đã sử dụng
    if rec:
        for i, item in enumerate(info):
            if i < len(rec):
                result = rec[i]
                item['hybrid_method'] = result[3]  # method
                item['cf_confidence'] = f"{result[4]:.3f}"  # cf_confidence
                
                # Thêm weights cho result đầu tiên nếu có
                if i == 0 and len(result) > 5:
                    item['cf_weight'] = f"{result[5]:.3f}"
                    item['content_weight'] = f"{result[6]:.3f}"
    
    return info

def system_recommend(selected_ids, cf_threshold=0.6):
    """
    System Recommend sử dụng Adaptive Hybrid cho user profile mới
    Kết hợp CF và Content-based với adaptive weighting
    """
    try:
        selected_ids = [int(i) for i in selected_ids]
        
        # Sử dụng HybridRecommender với adaptive_hybrid_user_from_likes
        rec = hybrid.adaptive_hybrid_user_from_likes(
            selected_movie_ids=selected_ids,
            content_based_rec=cb,
            top_k=10,
            cf_threshold=cf_threshold
        )
        
        # Convert kết quả để phù hợp với movieinfo format
        if rec:
            formatted_rec = []
            for result in rec:
                # Lấy 3 elements đầu tiên cho movieinfo: (movie_id, title, score)
                formatted_rec.append((result[0], result[1], result[2]))
        else:
            formatted_rec = []
        
        # Thêm thông tin về method và confidence vào response
        info = movieinfo(formatted_rec)
        
        # Thêm metadata về hybrid strategy đã sử dụng
        if rec:
            for i, item in enumerate(info):
                if i < len(rec):
                    result = rec[i]
                    item['hybrid_method'] = result[3]  # method
                    item['cf_confidence'] = f"{result[4]:.3f}"  # cf_confidence
                    
                    # Thêm weights cho result đầu tiên nếu có
                    if i == 0 and len(result) > 5:
                        item['cf_weight'] = f"{result[5]:.3f}"
                        item['content_weight'] = f"{result[6]:.3f}"
        
        return info
        
    except Exception as e:
        print(f"Error in system_recommend: {e}")
        raise

def cbitem(moviename):
    found, movie_data = MovieSearchUtils.normalize_and_find_movie(moviename, cb, cf, movie)
    if not found:
        raise ValueError(f"Không tìm thấy phim '{moviename}'.")
    
    resolved_title = movie_data['title']
    rec = cb.recommend_by_movie_fast(resolved_title, top_k=15)
    return movieinfo(rec)

def cbgenre(genres):
    rec = cb.recommend_by_genre(genres, top_k=15)
    return movieinfo(rec)

# --- WARM CACHE FOR ANCHOR MOVIES ---
def warm_anchor_cache():
    """Pre-fetch OMDB data for all anchor movies at startup"""
    print("Đang warm cache cho anchor movies...")
    anchor_movie_ids = []
    for title in ANCHOR_PRIMARY_GENRE.keys():
        movie_row = movie[movie["title"] == title]
        if not movie_row.empty:
            anchor_movie_ids.append((int(movie_row.iloc[0]["movieId"]), title, 0))
    
    if anchor_movie_ids:
        # Fetch với delay nhỏ để tránh rate limit
        import time
        for mid, title, score in anchor_movie_ids[:10]:  # Chỉ warm 10 movies đầu tiên
            movieinfo([(mid, title, score)])
            time.sleep(0.1)  # 100ms delay giữa mỗi request
        print(f"✅ Warmed cache for {min(10, len(anchor_movie_ids))} anchor movies")
    else:
        print("⚠️ No anchor movies found for cache warming")

# Call warm cache after movieinfo is defined (OPTIONAL - comment out nếu gặp vấn đề)
# warm_anchor_cache()
print("⚠️ Warm cache disabled - enable by uncommenting warm_anchor_cache()")

# --- 4. ROUTES (UI) ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend/<int:userid>")
def rcUI(userid):
    try:
        info = cfuser(userid)
        return render_template("recommend.html", mode="user", user_id=userid, movies=info)
    except Exception as e:
        return render_template("error.html", message=str(e))

@app.route("/similar/<moviename>")
def simUI(moviename):
    try:
        info = cfitem(moviename)
        return render_template("recommend.html", mode="item", query=moviename, movies=info)
    except ValueError as e:
        return render_template("error.html", message=str(e))

@app.route("/adaptive/<moviename>")
def adaptiveUI(moviename):
    """Route cho Adaptive Hybrid recommendation"""
    try:
        info = adaptive_hybrid_item(moviename)
        return render_template("recommend.html", mode="adaptive", query=moviename, movies=info)
    except ValueError as e:
        return render_template("error.html", message=str(e))

@app.route("/content/<moviename>")
def contentUI(moviename):
    try:
        info = cbitem(moviename)
        return render_template("recommend.html", mode="content", query=moviename, movies=info)
    except ValueError as e:
        return render_template("error.html", message=str(e))

@app.route("/genre/<genres>")
def genreUI(genres):
    try:
        info = cbgenre(genres)
        return render_template("recommend.html", mode="genre", query=genres, movies=info)
    except Exception as e:
        return render_template("error.html", message=str(e))

@app.route("/system")
def systemUI():
    """Route UI cho System Recommend"""
    return render_template("index.html", mode="system")

@app.route("/guidelines")
def guidelinesUI():
    try:
        guidelines_text = ContentBasedRec.get_search_guidelines()
        return render_template("guidelines.html", guidelines=guidelines_text)
    except Exception as e:
        return render_template("error.html", message=str(e))

# --- 5. API ROUTES ---
@app.route("/api/movies/sample")
def get_sample():
    import random, time

    refresh = request.args.get("refresh", "false") == "true"
    if refresh:
        random.seed(time.time())
    
    # 1.1 – Build anchors (20 movies)
    anchor_rows = movie[movie["title"].isin(ANCHOR_PRIMARY_GENRE.keys())]
    anchor_by_genre = {g: [] for g in ANCHOR_TARGETS}
    
    for _, r in anchor_rows.iterrows():
        g = ANCHOR_PRIMARY_GENRE.get(r["title"])
        if g in anchor_by_genre:
            anchor_by_genre[g].append(r)

    anchors = []
    used_ids = set()

    for genre, k in ANCHOR_TARGETS.items():
        pool = anchor_by_genre.get(genre, [])
        picks = random.sample(pool, k) if len(pool) >= k else pool
        
        for r in picks:
            anchors.append({
                "movieId": int(r["movieId"]),
                "title": r["title"]
            })
            used_ids.add(int(r["movieId"]))

    # Fill to 20
    remaining = anchor_rows[~anchor_rows["movieId"].isin(used_ids)]
    fill_needed = 20 - len(anchors)
    
    if fill_needed > 0 and not remaining.empty:
        for _, r in remaining.sample(n=min(fill_needed, len(remaining))).iterrows():
            anchors.append({
                "movieId": int(r["movieId"]),
                "title": r["title"]
            })
            used_ids.add(int(r["movieId"]))

    print(f"📊 [STEP 1.1] Anchors selected: {len(anchors)}/20 (expected: 20)")
    
    if len(anchors) < 16:
        print("⚠️ WARNING: Anchor thiếu - check movies.csv title match")

    # 1.2 – Build pool (exclude anchors)
    pool = movie[~movie["movieId"].isin(used_ids)].copy()
    pool["year"] = pool["title"].str.extract(r"\((\d{4})\)").astype(float)

    pool_old = pool[(pool["year"] >= 1980) & (pool["year"] < 2000)]
    pool_new = pool[pool["year"] >= 2000]

    # 1.3 – Select vintage (7–15 movies, NO OMDB)
    target_old = random.randint(7, 15)
    vintage = []
    genre_count = {}
    genre_cap = 5
    
    # Chỉ chọn từ mainstream genres, bỏ niche (War, Film-Noir, Western, Musical, Documentary)
    mainstream_genres = {'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 
                        'Fantasy', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'}

    for _, r in pool_old.sample(frac=1).iterrows():
        if len(vintage) >= target_old:
            break
        if pd.isna(r["genres"]):
            continue

        primary = r["genres"].split("|")[0]
        
        # Bỏ niche genres
        if primary not in mainstream_genres:
            continue
            
        if genre_count.get(primary, 0) >= genre_cap:
            continue

        vintage.append({
            "movieId": int(r["movieId"]),
            "title": r["title"]
        })
        genre_count[primary] = genre_count.get(primary, 0) + 1

    print(f"📊 [STEP 1.3] Vintage selected: {len(vintage)}/{target_old} (expected: 7-15, target: {target_old})")
    
    # 1.4 – Select modern with WEIGHTED random sampling (rating as weight)
    target_new = 30 - len(vintage)
    
    # Use cached avg_ratings instead of calculating each time
    pool_new_rated = pool_new.merge(avg_ratings_cache, on="movieId", how="left")
    pool_new_rated["avg_rating"].fillna(0, inplace=True)
    pool_new_rated["rating_count"].fillna(0, inplace=True)
    
    # Filter: only movies với >= 50 ratings
    pool_new_rated = pool_new_rated[pool_new_rated["rating_count"] >= 50]
    
    # WEIGHTED RANDOM: rating làm weight, normalize to probability
    pool_new_rated["weight"] = pool_new_rated["avg_rating"] ** 2  # Square để tăng bias cho rating cao
    total_weight = pool_new_rated["weight"].sum()
    pool_new_rated["prob"] = pool_new_rated["weight"] / total_weight if total_weight > 0 else 1.0 / len(pool_new_rated)
    
    modern = []
    selected_indices = set()
    
    while len(modern) < target_new and len(selected_indices) < len(pool_new_rated):
        # Weighted random sampling
        available = pool_new_rated[~pool_new_rated.index.isin(selected_indices)]
        if available.empty:
            break
            
        # Sample 1 row theo probability
        sampled = available.sample(n=1, weights="prob")
        idx = sampled.index[0]
        r = sampled.iloc[0]
        
        selected_indices.add(idx)
        
        if pd.isna(r["genres"]):
            continue

        primary = r["genres"].split("|")[0]
        
        # Bỏ niche genres
        if primary not in mainstream_genres:
            continue
            
        if genre_count.get(primary, 0) >= genre_cap:
            continue

        modern.append({
            "movieId": int(r["movieId"]),
            "title": r["title"]
        })
        genre_count[primary] = genre_count.get(primary, 0) + 1

    print(f"📊 [STEP 1.4] Modern selected: {len(modern)}/{target_new} (expected: 30-{len(vintage)}, target: {target_new})")
    
    # =====================================================
    # STEP 2 – Cố định danh sách 50 phim
    # =====================================================
    final_50 = anchors + vintage + modern
    
    print(f"📊 [STEP 2] Before ensure: {len(final_50)} = {len(anchors)} anchors + {len(vintage)} vintage + {len(modern)} modern")
    
    # Ensure exactly 50
    if len(final_50) < 50:
        needed = 50 - len(final_50)
        used_all_ids = {m["movieId"] for m in final_50}
        fallback = pool[~pool["movieId"].isin(used_all_ids)]
        
        if not fallback.empty:
            for _, r in fallback.sample(n=min(needed, len(fallback))).iterrows():
                final_50.append({
                    "movieId": int(r["movieId"]),
                    "title": r["title"]
                })
        print(f"⚠️ [STEP 2] Added {needed} fallback movies to reach 50")
    
    elif len(final_50) > 50:
        print(f"⚠️ [STEP 2] Trimming from {len(final_50)} to 50")
        final_50 = final_50[:50]

    print(f"✅ [STEP 2] Final count: {len(final_50)} (expected: 50)")

    # =====================================================
    # STEP 3 – Fetch OMDB cho đúng 50 phim
    # =====================================================
    final_tuples = [(m["movieId"], m["title"], 0) for m in final_50]
    final_movies = movieinfo(final_tuples)
    
    random.shuffle(final_movies)

    return jsonify({
        "movies": final_movies,
        "metadata": {
            "total": len(final_movies),
            "anchors": len(anchors),
            "vintage_1980_2000": len(vintage),
            "modern_2000_plus": len(modern),
            "modern_ratio": round(len(modern) / len(final_movies), 2) if final_movies else 0,
            "genre_distribution": dict(genre_count),
            "omdb_calls": len(final_50)  # Chỉ đúng 50 calls
        }
    })

@app.route("/api/recommend_personal", methods=["POST"])
def recommend_personal():
    """
    API endpoint chính cho Personal Recommendations
    Trả về 3 loại gợi ý: System (Adaptive Hybrid), CF thuần, Content-based thuần
    """
    data = request.json
    selected_ids = data.get("movieIds", [])
    if not selected_ids or len(selected_ids) < 3:
        return jsonify({
            "error": f"Vui lòng chọn ít nhất 3 phim để có được gợi ý chính xác. Bạn đã chọn {len(selected_ids)} phim."
        }), 400
    
    try:
        selected_ids = [int(i) for i in selected_ids]
        cf_threshold = data.get('cf_threshold', 0.6)
        
        # --- 1. System Recommend (Adaptive Hybrid) ---
        system_results = system_recommend(selected_ids, cf_threshold=cf_threshold)
        
        # --- 2. Collaborative Filtering thuần ---
        recs_cf = cf.recommend_from_likes(selected_ids, top_k=12)
        
        # --- 3. Content-Based thuần ---
        recs_cb = []
        per_movie_k = max(2, 12 // len(selected_ids)) 
        for mid in selected_ids:
            movie_row = movie[movie['movieId'] == mid]
            if not movie_row.empty:
                title = movie_row['title'].values[0]
                cb_matches = cb.recommend_by_movie_fast(title, top_k=per_movie_k)
                recs_cb.extend(cb_matches)
        
        # Loại bỏ các phim user đã chọn
        recs_cb = [r for r in recs_cb if r[0] not in selected_ids][:12]

        return jsonify({
            "system_results": system_results,  # Adaptive Hybrid (Primary)
            "cf_results": movieinfo(recs_cf),  # CF thuần (Comparison)
            "cb_results": movieinfo(recs_cb),  # Content thuần (Comparison)
            "metadata": {
                "method": "adaptive_hybrid_user_from_likes",
                "cf_threshold": cf_threshold,
                "selected_movies": len(selected_ids),
                "system_count": len(system_results),
                "cf_count": len(recs_cf),
                "cb_count": len(recs_cb)
            }
        })
    except Exception as e:
        print(f"Error in recommend_personal: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/rc/<int:userid>")
def rcapi(userid):
    try:
        return jsonify(cfuser(userid))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/sim/<path:moviename>")
def simapi(moviename):
    try:
        return jsonify(cfitem(moviename))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/adaptive/<path:moviename>")
def adaptiveapi(moviename):
    try:
        cf_threshold = request.args.get('cf_threshold', 0.6, type=float)
        info = adaptive_hybrid_item(moviename, cf_threshold=cf_threshold)
        return jsonify(info)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/content/<path:moviename>")
def contentapi(moviename):
    try:
        return jsonify(cbitem(moviename))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/genre/<path:genres>")
def genreapi(genres):
    try:
        return jsonify(cbgenre(genres))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/search/<path:moviename>")
def searchapi(moviename):
    try:
        candidates = cb.search_movies_interactive(moviename, max_results=10)
        if not candidates:
            return jsonify({"error": f"Không tìm thấy phim '{moviename}'"}), 404
        return jsonify({"query": moviename, "found": len(candidates), "candidates": candidates})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cache/stats")
def cache_stats():
    """API để xem thống kê cache"""
    with cache_stats_lock:
        hits = cache_stats_data['hits']
        misses = cache_stats_data['misses']
    
    total_requests = hits + misses
    hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
    
    return jsonify({
        "cache_size": len(omdb_cache),
        "total_hits": hits,
        "total_misses": misses,
        "hit_rate": f"{hit_rate:.1f}%",
        "cached_movies": list(omdb_cache.keys())[:20]  
    })

@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """API để xóa cache (thread-safe)"""
    with omdb_cache_lock:
        old_size = len(omdb_cache)
        omdb_cache.clear()
    return jsonify({
        "message": "Cache cleared successfully",
        "cleared_items": old_size
    })

# --- 6. START SERVER ---
if __name__ == "__main__":
    print("Các route đang hoạt động:")
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True)