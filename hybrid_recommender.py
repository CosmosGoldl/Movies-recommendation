import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


class HybridRecommender:
    """
    Adaptive Hybrid Recommender với intelligent weighting
    """
    
    def __init__(self, cf_model, datarating, datamovie):
        """
        Args:
            cf_model: ALSRec instance (collaborative filtering model)  
            datarating: DataFrame chứa ratings
            datamovie: DataFrame chứa thông tin movies
        """
        self.cf_model = cf_model
        self.datarating = datarating  
        self.datamovie = datamovie
    
    def calculate_cf_confidence(self, movie_id, sim_scores):
        """
        Tính độ tin cậy của CF recommendation dựa trên:
        - Số lượng ratings của phim
        - Chất lượng neighborhood (không chỉ mean/std)
        - Độ phổ biến của phim trong training data
        """
        # 1. Số lượng ratings (popularity confidence)
        movie_ratings_count = len(self.datarating[self.datarating['movieId'] == movie_id])
        max_ratings = self.datarating['movieId'].value_counts().max()
        popularity_conf = min(movie_ratings_count / (max_ratings * 0.1), 1.0)
        
        # 2. Chất lượng neighborhood - cải thiện từ mean_sim * (1 - std)
        if len(sim_scores) > 0:
            sim_scores_array = np.array(sim_scores)
            mean_sim = np.mean(sim_scores_array)
            std_sim = np.std(sim_scores_array)
            
            # Neighborhood quality: kết hợp nhiều yếu tố
            consistency_score = max(0, 1 - (std_sim / max(mean_sim, 0.1)))  # Tránh chia 0
            coverage_score = min(len(sim_scores) / 20, 1.0)  # Penalty nếu ít neighbors
            threshold_score = len([s for s in sim_scores if s > 0.3]) / max(len(sim_scores), 1)  # % neighbors có sim > threshold
            
            similarity_conf = mean_sim * 0.5 + consistency_score * 0.25 + coverage_score * 0.15 + threshold_score * 0.1
        else:
            similarity_conf = 0.0
        
        # 3. Confidence tổng hợp
        total_confidence = (popularity_conf * 0.3 + similarity_conf * 0.7)  # Tăng trọng số cho quality
        return min(total_confidence, 1.0)
    
    def _get_cf_recommendations(self, movie_id, movie_idx, top_k=20):
        """
        Lấy CF recommendations với true cosine similarity và filter phim gốc
        """
        cf_results = []
        cf_confidence = 0.0
        
        try:
            # Lấy item embeddings để tính cosine similarity thực tế
            item_factors = self.cf_model.model.item_factors
            query_item_vec = item_factors[movie_idx]
            
            # Tính cosine similarity với tất cả items
            similarities = cosine_similarity([query_item_vec], item_factors)[0]
            
            # Lấy top similar items (loại bỏ chính nó)
            similar_indices = np.argsort(similarities)[::-1]
            # Filter bỏ phim gốc
            similar_indices = [idx for idx in similar_indices if idx != movie_idx]
            
            pairs = [(idx, similarities[idx]) for idx in similar_indices[:top_k]]
            
            # Tính CF confidence
            cf_scores = [score for _, score in pairs]
            cf_confidence = self.calculate_cf_confidence(movie_id, cf_scores)
            
            # Convert to movie results và filter phim gốc
            for iid, score in pairs:
                if iid in self.cf_model.item and len(cf_results) < top_k:
                    mid = int(self.cf_model.item[iid])
                    # Double check filter phim gốc với debug
                    if mid != movie_id:
                        title = self.datamovie.loc[self.datamovie['movieId'] == mid, 'title'].values[0]
                        cf_results.append((mid, str(title), float(score)))
                    else:
                        print(f" Filtered original movie: {mid} == {movie_id}")
                        
        except Exception as e:
            print(f"CF failed: {e}")
            cf_confidence = 0.0
            
        return cf_results, cf_confidence
    
    def _get_content_recommendations(self, movie_title, content_based_rec, movie_id, top_k=20):
        """
        Lấy Content-based recommendations và filter phim gốc
        """
        content_results = []
        if content_based_rec is not None:
            try:
                content_results = content_based_rec.recommend_by_movie(movie_title, top_k)
                # Format to match CF output: (movie_id, title, score) và filter phim gốc
                content_results = [(r[0], r[1], r[2]) for r in content_results if r[0] != movie_id]
            except Exception as e:
                print(f"Content-based failed: {e}")
        return content_results
    
    def _calculate_adaptive_weights(self, cf_confidence, cf_threshold=0.6, content_weight_min=0.2, content_weight_max=0.8):
        """
        Tính adaptive weights với continuous sigmoid function và sử dụng cf_threshold
        Đảm bảo CF luôn có minimum weight 0.15 để không bị "chết hẳn"
        """
        # Sigmoid-based continuous weight calculation
        def sigmoid(x, center=0.5, steepness=8):
            return 1 / (1 + np.exp(-steepness * (x - center)))
        
        # Sử dụng cf_threshold để điều chỉnh sigmoid center
        sigmoid_center = cf_threshold  # Threshold làm center point
        
        # Content weight tăng khi CF confidence thấp
        normalized_conf = max(0, min(1, cf_confidence))  # Clamp to [0,1]
        # Invert confidence: conf cao -> content weight thấp
        inverted_conf = 1 - normalized_conf
        
        # Smooth transition với sigmoid, sử dụng cf_threshold làm center
        content_weight_raw = sigmoid(inverted_conf, center=1-sigmoid_center, steepness=6)
        # Scale vào range [min, max]
        content_weight = content_weight_min + (content_weight_max - content_weight_min) * content_weight_raw
        
        # Đảm bảo CF luôn có minimum weight 0.15 để không bị "chết hẳn"
        cf_min_weight = 0.15
        content_weight = min(content_weight, 1.0 - cf_min_weight)
        cf_weight = 1.0 - content_weight
        
        # Determine method based on final weights và cf_threshold
        if cf_confidence >= cf_threshold and content_weight < 0.4:
            method = "cf_dominant"
        elif cf_confidence < cf_threshold * 0.5 and content_weight > 0.6:
            method = "content_dominant"
        else:
            method = "balanced_hybrid"
            
        return cf_weight, content_weight, method
    
    def _normalize_scores(self, results):
        """Helper function để normalize scores về [0,1] gọn hơn"""
        if not results:
            return []
        scores = [score for _, _, score in results]
        score_min, score_max = min(scores), max(scores)
        score_range = max(score_max - score_min, 1e-6)  # Tránh chia 0
        return [(mid, title, (score - score_min) / score_range) for mid, title, score in results]
    
    def _normalize_and_combine_scores(self, cf_results, content_results, cf_weight, content_weight):
        """
        Normalize scores về [0,1] và combine với weights - phiên bản gọn
        """
        # Normalize bằng helper function
        cf_normalized = self._normalize_scores(cf_results)
        content_normalized = self._normalize_scores(content_results)
        
        combined_scores = {}
        
        # Thêm CF results
        for mid, title, norm_score in cf_normalized:
            combined_scores[mid] = {
                'title': title,
                'cf_score': norm_score * cf_weight,
                'content_score': 0.0,
                'total_score': norm_score * cf_weight
            }
        
        # Thêm Content results
        for mid, title, norm_score in content_normalized:
            if mid in combined_scores:
                combined_scores[mid]['content_score'] = norm_score * content_weight
                combined_scores[mid]['total_score'] += norm_score * content_weight
            else:
                combined_scores[mid] = {
                    'title': title,
                    'cf_score': 0.0,
                    'content_score': norm_score * content_weight,
                    'total_score': norm_score * content_weight
                }
                
        return combined_scores
    
    def adaptive_hybrid_simitem(self, movie_name, content_based_rec=None, top_k=10, 
                               cf_threshold=0.6, content_weight_min=0.2, content_weight_max=0.8):
        """
        Adaptive Hybrid cho simitem - cải thiện với:
        - Score normalization về cùng scale [0,1]
        - Continuous weight selection (không dùng if-else)
        - True cosine similarity cho ALS
        - Filter phim gốc khỏi kết quả
        - Neighborhood quality assessment
        
        Args:
            movie_name: Tên phim cần tìm similar
            content_based_rec: Instance của ContentBasedRec class  
            top_k: Số lượng gợi ý trả về
            cf_threshold: Ngưỡng confidence để quyết định hybrid strategy
            content_weight_min: Trọng số tối thiểu của content-based
            content_weight_max: Trọng số tối đa của content-based
        
        Returns:
            List of tuples: (movie_id, title, score, method_used, cf_confidence)
        """
        if self.datamovie is None:
            raise RuntimeError("Cần datamovie để tìm phim")
        
        # Tìm phim gốc
        match = self.datamovie[self.datamovie['title'].str.contains(movie_name, case=False, regex=False)]
        if match.empty:
            all_titles = self.datamovie['title'].tolist()
            close = get_close_matches(movie_name, all_titles, n=1, cutoff=0.6)
            if close:
                match = self.datamovie[self.datamovie['title'] == close[0]]
        
        if match.empty:
            raise ValueError(f"Không tìm thấy phim gần đúng với: {movie_name}")
        
        movie_id = int(match.iloc[0]['movieId'])
        movie_title = match.iloc[0]['title']
        
        # Check if movie exists in CF model - Cold start handling
        cf_available = movie_id in self.cf_model.itemmap
        
        if not cf_available:
            print(f" Cold-start item: {movie_name} - Fallback to content-only")
            # Fallback: Content-only cho cold-start items
            if content_based_rec is not None:
                try:
                    content_results = content_based_rec.recommend_by_movie(movie_title, top_k)
                    content_filtered = [(r[0], r[1], r[2]) for r in content_results if r[0] != movie_id]
                    final_results = []
                    for mid, title, score in content_filtered[:top_k]:
                        final_results.append((mid, title, score, "content_only", 0.0))
                    
                    # Thêm weights info cho result đầu tiên để consistent với format
                    if final_results:
                        first = final_results[0]
                        final_results[0] = (first[0], first[1], first[2], first[3], first[4], 0.0, 1.0)  # CF=0, Content=1
                    
                    print(f" Content-only fallback for '{movie_name}': {len(final_results)} results")
                    return final_results
                except Exception as e:
                    print(f"Content-based fallback failed: {e}")
                    return []  # Không có gợi ý nào
            else:
                print(f"No content-based available for cold-start: {movie_name}")
                return []
        
        movie_idx = self.cf_model.itemmap[movie_id]
        
        # 1. Lấy CF recommendations
        cf_results, cf_confidence = self._get_cf_recommendations(movie_id, movie_idx, top_k * 2)
        
        # 2. Lấy Content-based recommendations
        content_results = self._get_content_recommendations(movie_title, content_based_rec, movie_id, top_k * 2)
        
        # 3. Tính adaptive weights với cf_threshold
        cf_weight, content_weight, method = self._calculate_adaptive_weights(
            cf_confidence, cf_threshold, content_weight_min, content_weight_max
        )
        
        # 4. Normalize và combine scores
        combined_scores = self._normalize_and_combine_scores(cf_results, content_results, cf_weight, content_weight)
        
        # 5. Sort và return top_k với final filter
        final_results = []
        sorted_items = sorted(combined_scores.items(), 
                            key=lambda x: x[1]['total_score'], reverse=True)
        
        for mid, data in sorted_items:
            # Final safety filter để đảm bảo 100% không có phim gốc
            if mid != movie_id and len(final_results) < top_k:
                final_results.append((
                    mid, 
                    data['title'], 
                    data['total_score'],
                    method,
                    cf_confidence
                ))
            elif mid == movie_id:
                print(f" Final filter blocked original movie: {mid} - {data['title']}")
        
        # Thêm thông tin weights vào kết quả đầu tiên cho API logging
        if final_results:
            # Rebuild result đầu tiên với weights info
            first = final_results[0]
            final_results[0] = (first[0], first[1], first[2], first[3], first[4], cf_weight, content_weight)
        
        print(f"Adaptive Hybrid for '{movie_name}': CF_confidence={cf_confidence:.3f}, "
              f"Method={method}, CF_weight={cf_weight:.3f}, Content_weight={content_weight:.3f}, "
              f"CF_threshold={cf_threshold:.3f}, CF_results={len(cf_results)}, Content_results={len(content_results)}")
        
        return final_results  # Format: (movie_id, title, score, method, cf_confidence, [cf_weight, content_weight])

    def calculate_user_cf_confidence(self, selected_movie_ids, cf_scores):
        """
        Tính độ tin cậy của CF recommendation cho user dựa trên:
        - Số lượng phim đã like (coverage)
        - Độ phổ biến trung bình của các phim đã like
        - Chất lượng của CF scores
        """
        # 1. Coverage confidence - số lượng phim known trong training data
        known_movies = [mid for mid in selected_movie_ids if mid in self.cf_model.itemmap]
        coverage_conf = min(len(known_movies) / max(len(selected_movie_ids), 1), 1.0)
        
        # 2. Popularity confidence - độ phổ biến trung bình của phim đã like
        if known_movies:
            avg_ratings_count = 0
            max_ratings = self.datarating['movieId'].value_counts().max()
            
            for mid in known_movies:
                movie_ratings_count = len(self.datarating[self.datarating['movieId'] == mid])
                avg_ratings_count += movie_ratings_count
            
            avg_ratings_count /= len(known_movies)
            popularity_conf = min(avg_ratings_count / (max_ratings * 0.1), 1.0)
        else:
            popularity_conf = 0.0
        
        # 3. CF scores quality
        if len(cf_scores) > 0:
            scores_array = np.array(cf_scores)
            mean_score = np.mean(scores_array)
            std_score = np.std(scores_array)
            
            # Score quality: kết hợp mean và consistency
            mean_quality = min(mean_score / 2.0, 1.0)  # Normalize assuming max CF score ~2
            consistency_quality = max(0, 1 - (std_score / max(mean_score, 0.1)))
            coverage_quality = min(len(cf_scores) / 20, 1.0)  # Penalty nếu ít recommendations
            
            scores_conf = mean_quality * 0.5 + consistency_quality * 0.3 + coverage_quality * 0.2
        else:
            scores_conf = 0.0
        
        # 4. Confidence tổng hợp
        total_confidence = (coverage_conf * 0.4 + popularity_conf * 0.3 + scores_conf * 0.3)
        return min(total_confidence, 1.0)

    def _get_cf_user_recommendations(self, selected_movie_ids, top_k=20):
        """
        Lấy CF recommendations cho user dựa trên danh sách phim đã like
        """
        cf_results = []
        cf_confidence = 0.0
        
        try:
            # Sử dụng recommend_from_likes của CF model
            cf_results_raw = self.cf_model.recommend_from_likes(selected_movie_ids, top_k)
            
            # Format to match expected output và filter các phim đã like
            cf_results = []
            cf_scores = []
            
            for mid, title, score in cf_results_raw:
                if mid not in selected_movie_ids:  # Filter phim đã like
                    cf_results.append((mid, title, float(score)))
                    cf_scores.append(float(score))
            
            # Tính CF confidence
            cf_confidence = self.calculate_user_cf_confidence(selected_movie_ids, cf_scores)
            
        except Exception as e:
            print(f"User CF failed: {e}")
            cf_confidence = 0.0
            
        return cf_results, cf_confidence

    def _get_content_user_recommendations(self, selected_movie_ids, content_based_rec, top_k=20):
        """
        Lấy Content-based recommendations cho user dựa trên danh sách phim đã like
        """
        content_results = []
        
        if content_based_rec is not None:
            try:
                # Lấy recommendations từ nhiều phim đã like và aggregate
                all_recommendations = {}
                
                for movie_id in selected_movie_ids:
                    # Tìm title của movie_id
                    movie_match = self.datamovie[self.datamovie['movieId'] == movie_id]
                    if movie_match.empty:
                        continue
                        
                    movie_title = movie_match.iloc[0]['title']
                    
                    try:
                        # Lấy recommendations cho từng phim
                        movie_recs = content_based_rec.recommend_by_movie(movie_title, top_k * 2)
                        
                        for mid, title, score in movie_recs:
                            if mid not in selected_movie_ids:  # Filter phim đã like
                                if mid in all_recommendations:
                                    # Aggregate scores bằng cách lấy max (có thể thay bằng mean)
                                    all_recommendations[mid] = (
                                        title, 
                                        max(all_recommendations[mid][1], score)
                                    )
                                else:
                                    all_recommendations[mid] = (title, score)
                                    
                    except Exception as e:
                        print(f"Content recommendation failed for movie {movie_title}: {e}")
                        continue
                
                # Convert dictionary to list và sort
                content_results = [
                    (mid, data[0], data[1]) 
                    for mid, data in all_recommendations.items()
                ]
                content_results = sorted(content_results, key=lambda x: x[2], reverse=True)[:top_k]
                
            except Exception as e:
                print(f"Content-based user recommendations failed: {e}")
                
        return content_results

    def adaptive_hybrid_user_from_likes(self, selected_movie_ids, content_based_rec=None, top_k=10,
                                      cf_threshold=0.6, content_weight_min=0.2, content_weight_max=0.8):
        """
        Adaptive Hybrid recommendations cho user mới dựa trên danh sách phim đã like
        Kết hợp CF (recommend_from_likes) và Content-based với adaptive weighting
        
        Args:
            selected_movie_ids: List các movieId mà user đã like
            content_based_rec: Instance của ContentBasedRec class
            top_k: Số lượng gợi ý trả về
            cf_threshold: Ngưỡng confidence để quyết định hybrid strategy
            content_weight_min: Trọng số tối thiểu của content-based
            content_weight_max: Trọng số tối đa của content-based
            
        Returns:
            List of tuples: (movie_id, title, score, method_used, cf_confidence)
        """
        if not selected_movie_ids:
            raise ValueError("Cần ít nhất 1 phim đã like để tạo gợi ý")
            
        if self.datamovie is None:
            raise RuntimeError("Cần datamovie để tìm thông tin phim")
        
        # Validate selected movies exist in database
        valid_movie_ids = []
        for mid in selected_movie_ids:
            if not self.datamovie[self.datamovie['movieId'] == mid].empty:
                valid_movie_ids.append(mid)
            else:
                print(f" Warning: Movie ID {mid} không tồn tại trong database")
                
        if not valid_movie_ids:
            raise ValueError("Không có phim hợp lệ nào trong danh sách đã like")
        
        selected_movie_ids = valid_movie_ids
        
        # Check cold-start situation
        known_in_cf = [mid for mid in selected_movie_ids if mid in self.cf_model.itemmap]
        cold_start_ratio = 1 - (len(known_in_cf) / len(selected_movie_ids))
        
        print(f" User profile: {len(selected_movie_ids)} liked movies, "
              f"{len(known_in_cf)} known in CF, cold-start ratio: {cold_start_ratio:.2f}")
        
        if cold_start_ratio > 0.95:  # Chỉ khi >95% phim không có trong CF mới fallback hoàn toàn
            print(" Extreme cold-start ratio - Fallback to content-only")
            # Fallback: Content-only cho extreme cold-start users
            if content_based_rec is not None:
                try:
                    content_results = self._get_content_user_recommendations(
                        selected_movie_ids, content_based_rec, top_k
                    )
                    final_results = []
                    for mid, title, score in content_results[:top_k]:
                        final_results.append((mid, title, score, "content_only", 0.0))
                    
                    # Thêm weights info cho result đầu tiên
                    if final_results:
                        first = final_results[0]
                        final_results[0] = (first[0], first[1], first[2], first[3], first[4], 0.0, 1.0)
                    
                    print(f"Content-only fallback: {len(final_results)} results")
                    return final_results
                    
                except Exception as e:
                    print(f"Content-based fallback failed: {e}")
                    return []
            else:
                print("No content-based available for cold-start user")
                return []
        elif cold_start_ratio > 0.7:  # High cold-start: giảm CF confidence nhưng vẫn dùng hybrid
            print(f" High cold-start ratio ({cold_start_ratio:.2f}) - CF confidence will be reduced")
        
        # 1. Lấy CF recommendations
        cf_results, cf_confidence = self._get_cf_user_recommendations(selected_movie_ids, top_k * 2)
        
        # Điều chỉnh CF confidence dựa trên cold-start ratio
        if cold_start_ratio > 0.7:
            # Giảm CF confidence nhưng không về 0 hoàn toàn
            cf_confidence = cf_confidence * (1 - cold_start_ratio * 0.5)  # Giảm tối đa 50%
            print(f" CF confidence adjusted for cold-start: {cf_confidence:.3f}")
        
        # 2. Lấy Content-based recommendations  
        content_results = self._get_content_user_recommendations(
            selected_movie_ids, content_based_rec, top_k * 2
        )
        
        # 3. Tính adaptive weights
        cf_weight, content_weight, method = self._calculate_adaptive_weights(
            cf_confidence, cf_threshold, content_weight_min, content_weight_max
        )
        
        # 4. Normalize và combine scores
        combined_scores = self._normalize_and_combine_scores(cf_results, content_results, cf_weight, content_weight)
        
        # 5. Sort và return top_k
        final_results = []
        sorted_items = sorted(combined_scores.items(), 
                            key=lambda x: x[1]['total_score'], reverse=True)
        
        for mid, data in sorted_items:
            if len(final_results) < top_k:
                final_results.append((
                    mid,
                    data['title'],
                    data['total_score'],
                    method,
                    cf_confidence
                ))
        
        # Thêm thông tin weights vào kết quả đầu tiên
        if final_results:
            first = final_results[0]
            final_results[0] = (first[0], first[1], first[2], first[3], first[4], cf_weight, content_weight)
        
        # Logging
        liked_titles = []
        for mid in selected_movie_ids[:3]:  # Show first 3 liked movies
            title_match = self.datamovie[self.datamovie['movieId'] == mid]
            if not title_match.empty:
                liked_titles.append(title_match.iloc[0]['title'])
        
        print(f"Adaptive Hybrid User (liked: {', '.join(liked_titles)}...): "
              f"CF_confidence={cf_confidence:.3f}, Method={method}, "
              f"CF_weight={cf_weight:.3f}, Content_weight={content_weight:.3f}, "
              f"CF_results={len(cf_results)}, Content_results={len(content_results)}")
        
        return final_results  # Format: (movie_id, title, score, method, cf_confidence, [cf_weight, content_weight])