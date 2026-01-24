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
OMDB_API_KEY = "e54bd3ba"

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
                    omdb = requests.get(url, timeout=5).json()
                    # Lưu kết quả vào cache (thay thế _FETCHING)
                    if omdb.get('Response') == 'True':
                        with omdb_cache_lock:
                            omdb_cache[imdbid] = omdb
                    else:
                        # Remove sentinel nếu fetch failed
                        with omdb_cache_lock:
                            omdb_cache.pop(imdbid, None)
                except:
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
    try:
        import random
        import time
        
        # Kiểm tra parameter refresh để thay đổi seed cho random
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        if refresh:
            random.seed(int(time.time()))
        
        # BƯỚC 1: Chọn NHIỀU phim từ database (40-50 phim) - KHÔNG GỌI OMDB
        movie_copy = movie.copy()
        movie_copy['year'] = movie_copy['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        # Ưu tiên phim từ 2000 trở về sau
        modern_movies = movie_copy[(movie_copy['year'] >= 2000) & (movie_copy['year'].notna())]
        classic_movies = movie_copy[(movie_copy['year'] < 2000) & (movie_copy['year'] >= 1980) & (movie_copy['year'].notna())]
        
        # Đa dạng genre: chia thành 3 nhóm
        mainstream_genres = ['Action', 'Comedy', 'Drama', 'Thriller']
        popular_genres = ['Romance', 'Adventure', 'Sci-Fi', 'Horror']
        niche_genres = ['Crime', 'Animation', 'Fantasy', 'Mystery', 'Documentary', 'War']
        
        candidate_movies = []
        seen_titles = set()
        
        # Lấy 3-4 phim cho mỗi mainstream genre (có nhiều lựa chọn để lọc sau)
        for genre in mainstream_genres:
            genre_movies = modern_movies[modern_movies['genres'].str.contains(genre, na=False, case=False)]
            if not genre_movies.empty:
                sample_size = min(4, len(genre_movies))
                genre_sample = genre_movies.sample(n=sample_size)
                
                for _, row in genre_sample.iterrows():
                    if row['title'] not in seen_titles:
                        seen_titles.add(row['title'])
                        # Fix #2: primary_genre là genre đầu tiên trong list thực tế
                        all_genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
                        primary = all_genres[0] if all_genres else genre
                        candidate_movies.append({
                            'movieId': row['movieId'],
                            'title': row['title'],
                            'year': row['year'],
                            'genres': row['genres'],
                            'primary_genre': primary,  # Dùng genre thực tế của phim
                            'search_genre': genre,  # Genre dùng để search
                            'score': 1.0
                        })
        
        # Lấy 2-3 phim cho mỗi popular genre
        for genre in popular_genres:
            genre_movies = modern_movies[modern_movies['genres'].str.contains(genre, na=False, case=False)]
            if not genre_movies.empty:
                sample_size = min(3, len(genre_movies))
                genre_sample = genre_movies.sample(n=sample_size)
                
                for _, row in genre_sample.iterrows():
                    if row['title'] not in seen_titles:
                        seen_titles.add(row['title'])
                        all_genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
                        primary = all_genres[0] if all_genres else genre
                        candidate_movies.append({
                            'movieId': row['movieId'],
                            'title': row['title'],
                            'year': row['year'],
                            'genres': row['genres'],
                            'primary_genre': primary,
                            'search_genre': genre,
                            'score': 0.8
                        })
        
        # Lấy 1-2 phim cho niche genres
        for genre in niche_genres:
            genre_movies = modern_movies[modern_movies['genres'].str.contains(genre, na=False, case=False)]
            if not genre_movies.empty and random.random() < 0.7:  # 70% chance
                genre_sample = genre_movies.sample(n=min(2, len(genre_movies)))
                for _, row in genre_sample.iterrows():
                    if row['title'] not in seen_titles:
                        seen_titles.add(row['title'])
                        all_genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
                        primary = all_genres[0] if all_genres else genre
                        candidate_movies.append({
                            'movieId': row['movieId'],
                            'title': row['title'],
                            'year': row['year'],
                            'genres': row['genres'],
                            'primary_genre': primary,
                            'search_genre': genre,
                            'score': 0.6
                        })
        
        # Thêm classics (3-5 phim)
        if not classic_movies.empty:
            classic_sample = classic_movies.sample(n=min(5, len(classic_movies)))
            for _, row in classic_sample.iterrows():
                if row['title'] not in seen_titles:
                    seen_titles.add(row['title'])
                    all_genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
                    primary = all_genres[0] if all_genres else 'Drama'
                    candidate_movies.append({
                        'movieId': row['movieId'],
                        'title': row['title'],
                        'year': row['year'],
                        'genres': row['genres'],
                        'primary_genre': primary,
                        'search_genre': 'Classic',
                        'score': 0.7
                    })
        
        # BƯỚC 2: RANK và LỌC xuống còn 20 phim tốt nhất 
        # Shuffle để tránh bias thứ tự
        random.shuffle(candidate_movies)
        
        # Đảm bảo đa dạng genre trong 20 phim được chọn
        selected_candidates = []
        selected_ids = set()  
        genre_counts = {}
        max_per_genre = 3  # Tối đa 3 phim/genre để đa dạng
        
        # Pass 1: Ưu tiên đa dạng genre
        for candidate in candidate_movies:
            genre = candidate['primary_genre']
            movie_id = candidate['movieId']
            if genre_counts.get(genre, 0) < max_per_genre and len(selected_candidates) < 20:
                selected_candidates.append(candidate)
                selected_ids.add(movie_id)
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Pass 2: Nếu chưa đủ 20, thêm phim còn lại
        for candidate in candidate_movies:
            movie_id = candidate['movieId']
            if movie_id not in selected_ids and len(selected_candidates) < 20:
                selected_candidates.append(candidate)
                selected_ids.add(movie_id)
        
        # BƯỚC 3: CHỈ GỌI OMDB cho 20 phim đã chọn (LAZY FETCH)
        movie_tuples = [(c['movieId'], c['title'], 0) for c in selected_candidates]
        movie_details = movieinfo(movie_tuples)  
        
        # Parse IMDb rating an toàn
        def safe_parse_rating(rating_str):
            """Parse IMDb rating, handle formats: '7.8', '7.8/10', 'N/A'"""
            try:
                if rating_str and rating_str not in ["N/A", ""]:
                    # Fix #3: Handle format "7.8/10"
                    if '/' in rating_str:
                        rating_str = rating_str.split('/')[0].strip()
                    rating = float(rating_str)
                    if 0 <= rating <= 10:
                        return rating
            except (ValueError, TypeError, AttributeError):
                pass
            return None
        
        # Cho phép phim không có poster - sort tất cả theo rating
        for movie_detail in movie_details:
            rating = safe_parse_rating(movie_detail.get('imdbRating'))
            movie_detail['_rating'] = rating if rating else 0
            # Bonus điểm cho phim có poster
            has_poster = movie_detail.get('poster') and movie_detail['poster'] not in ["N/A", ""]
            if has_poster:
                movie_detail['_rating'] += 0.5  # Bonus nhỏ cho phim có poster
        
        # Sort theo rating (bao gồm cả poster và không poster)
        movie_details.sort(key=lambda x: x.get('_rating', 0), reverse=True)
        final_movies = movie_details[:20]
        random.shuffle(final_movies)
        
        # Remove temporary rating field
        for m in final_movies:
            m.pop('_rating', None)
        
        # Genre distribution thực tế
        actual_genre_counts = {}
        for m in final_movies:
            movie_id = m['movieId']
            movie_row = movie[movie['movieId'] == movie_id]
            if not movie_row.empty:
                genres = movie_row['genres'].values[0]
                if isinstance(genres, str):
                    for genre in genres.split('|'):
                        genre = genre.strip()
                        actual_genre_counts[genre] = actual_genre_counts.get(genre, 0) + 1
        
        # Fix #4: Tính modern_ratio chuẩn hơn
        def is_modern_movie(year_str):
            """Check if movie is from 2000 or later"""
            try:
                if not year_str or year_str == 'N/A':
                    return False
                # Handle various formats: "2010", "2010-2012", etc.
                year = year_str.split('-')[0].strip()
                return int(year) >= 2000
            except (ValueError, AttributeError):
                return False
        
        modern_count = sum(1 for m in final_movies if is_modern_movie(m.get('year')))
        modern_ratio = modern_count / len(final_movies) if final_movies else 0
        
        # Fix #1: Get actual cache stats
        with cache_stats_lock:
            current_hits = cache_stats_data['hits']
            current_misses = cache_stats_data['misses']
            # Reset for next request tracking
            cache_stats_data['hits'] = 0
            cache_stats_data['misses'] = 0
        
        return jsonify({
            "movies": final_movies,
            "metadata": {
                "total_movies": len(final_movies),
                "candidates_evaluated": len(candidate_movies),
                "api_calls": current_misses,  
                "cache_hits": current_hits,
                "cache_total": len(omdb_cache),
                "genre_distribution": actual_genre_counts,
                "modern_ratio": round(modern_ratio, 2)
            }
        })
    except Exception as e:
        print(f"Error in get_sample: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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