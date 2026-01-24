import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import time
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f" {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class ContentBasedRec:
    def __init__(self, movies_df):
        """
        Content-based recommendation system optimized for Flask and Hybrid systems
        Uses vectorized operations for speed with all movies
        """
        start_time = time.time()
        
        print(f" Processing ALL {len(movies_df)} movies with vectorized operations")
        self.movies = movies_df.copy()
        self.movies['genres'] = self.movies['genres'].fillna('')
        self._title_cache = {}  # Cache for movie title lookups
        
        # Preprocessing
        self._preprocess_data()
        self._prepare_vectorizers()
        
        total_time = time.time() - start_time
        print(f" System initialized in {total_time:.2f} seconds with {len(self.movies)} movies")

    def _extract_franchise_name(self, title):
        """
        Extract franchise base name - IMPROVED logic with less regex
        """
        title_lower = title.lower().strip()
        
        # Remove year and common noise
        title_clean = re.sub(r'\(\d{4}\)', '', title_lower).strip()
        title_clean = re.sub(r'\b(the|a|an)\s+', '', title_clean)
        
        # Method 1: Colon-based franchise (most reliable)
        if ':' in title_clean:
            base = title_clean.split(':')[0].strip()
            if len(base) > 3 and not any(word in base for word in ['part', 'chapter', 'episode']):
                return base
        
        # Method 2: Number at end (but be more selective)
        number_match = re.search(r'^(.+?)\s+([ivx]+|\d+)$', title_clean)
        if number_match:
            base = number_match.group(1).strip()
            number = number_match.group(2)
            # Only if base is long enough and number looks like sequel
            if len(base) > 4 and (number.isdigit() or number in ['ii', 'iii', 'iv', 'v']):
                return base
        
        # Method 3: Subtitle patterns with keywords
        keyword_patterns = [
            r'^(.+?)\s+part\s+\w+',
            r'^(.+?)\s+chapter\s+\w+',
            r'^(.+?)\s+episode\s+\w+'
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, title_clean)
            if match:
                base = match.group(1).strip()
                if len(base) > 4:
                    return base
        
        return None
        
    def _preprocess_data(self):
        """Preprocess movie data"""
        # Clean genres and split into lists
        self.movies['genres_list'] = self.movies['genres'].apply(
            lambda x: [genre.strip().lower() for genre in x.split('|')] if x else []
        )
        
        # Extract year from title
        self.movies['year'] = self.movies['title'].apply(self._extract_year)
        
        # Clean title (remove year)
        self.movies['clean_title'] = self.movies['title'].apply(self._clean_title)
        
        # PRE-COMPUTE normalized titles for search optimization
        self.movies['normalized_title'] = self.movies['title'].apply(self.normalize_movie_title)
        
        # Create genre text for TF-IDF
        self.movies['genre_text'] = self.movies['genres_list'].apply(lambda x: ' '.join(x))
        
        # Build genre vocabulary for fast lookup
        self.all_genres = set()
        for genre_list in self.movies['genres_list']:
            self.all_genres.update(genre_list)
        self.all_genres = sorted(list(self.all_genres))
        
        # Create optimized movie index for O(1) lookup (handle duplicate titles)
        self.title_to_index = defaultdict(list)
        self.title_fuzzy_index = {}  # For fuzzy search optimization
        
        for idx, row in self.movies.iterrows():
            title_clean = row['clean_title'].lower().strip()
            title_original = row['title'].lower().strip()
            
            # Exact match index
            self.title_to_index[title_clean].append(idx)
            
            # Fuzzy search index (key words for faster partial matching)
            words = title_clean.split()
            for word in words:
                if len(word) > 2:  # Skip very short words
                    if word not in self.title_fuzzy_index:
                        self.title_fuzzy_index[word] = []
                    self.title_fuzzy_index[word].append((idx, title_clean))
            
    def _extract_year(self, title):
        """Extract year from movie title"""
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else None
        
    def _clean_title(self, title):
        """Remove year from title"""
        return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
        
    def _prepare_vectorizers(self):
        """Prepare TF-IDF vectorizers - OPTIMIZED for speed and memory"""
        # Title vectorizer - optimized
        self.title_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 1),
            max_features=1500,  # Fixed optimal value
            lowercase=True,
            min_df=1,  # Don't exclude rare movies
            dtype=np.float32  # Memory optimization
        )
        
        # Genre vectorizer - OPTIMIZED (no lambda tokenizer)
        self.genre_vectorizer = TfidfVectorizer(
            vocabulary=self.all_genres,  # Pre-defined vocabulary
            lowercase=False,
            binary=True,  # Genres are binary features
            dtype=np.float32
        )
        
        print(f" Preparing vectorizers for {len(self.movies)} movies...")
        start_time = time.time()
        
        # Fit vectorizers with memory optimization
        self.title_tfidf_matrix = self.title_vectorizer.fit_transform(self.movies['clean_title'])
        self.genre_tfidf_matrix = self.genre_vectorizer.fit_transform(self.movies['genre_text'])
        
        # Convert to CSR format for faster operations
        self.title_tfidf_matrix = self.title_tfidf_matrix.tocsr()
        self.genre_tfidf_matrix = self.genre_tfidf_matrix.tocsr()
        
        load_time = time.time() - start_time
        print(f" Vectorizers ready in {load_time:.1f}s - {len(self.movies)} movies loaded")
        
    def normalize_movie_title(self, title):
        """
        Comprehensive movie title normalization for better search results
        Handles various formats and common issues
        """
        if not title or not isinstance(title, str):
            return ""
            
        # Convert to lowercase and strip
        title = title.lower().strip()
        
        # Remove extra spaces
        title = re.sub(r'\s+', ' ', title)
        
        # Handle "the", "a", "an" articles - move from beginning to end with comma
        article_patterns = [
            (r'^the\s+(.+)$', r'\1, the'),
            (r'^a\s+(.+)$', r'\1, a'), 
            (r'^an\s+(.+)$', r'\1, an')
        ]
        
        for pattern, replacement in article_patterns:
            if re.match(pattern, title):
                title = re.sub(pattern, replacement, title)
                break
        
        # Handle year formats - extract and normalize
        # Remove year from title for searching (we'll handle year matching separately)
        title_without_year = re.sub(r'\s*[\(\[]?(19|20)\d{2}[\)\]]?\s*$', '', title).strip()
        
        return title_without_year
    
    def extract_year_from_title(self, title):
        """
        Extract year from movie title if present
        """
        if not title:
            return None
            
        # Look for year patterns
        year_match = re.search(r'\b(19|20)(\d{2})\b', title)
        if year_match:
            return int(year_match.group(1) + year_match.group(2))
        return None
    
    def find_movie_candidates(self, search_title):
        """
        OPTIMIZED: Find movie candidates with vectorized operations instead of O(N) loop
        Returns list of (index, title, score, year) sorted by relevance
        """
        if not search_title:
            return []
            
        search_normalized = self.normalize_movie_title(search_title)
        search_year = self.extract_year_from_title(search_title)
        
        # Use pre-computed normalized titles for vectorized comparison
        normalized_titles = self.movies['normalized_title']
        movie_years = self.movies['year']
        
        candidates = []
        
        # VECTORIZED OPERATIONS - much faster than iterrows()
        
        # 1. Exact matches (highest priority)
        exact_mask = (normalized_titles == search_normalized)
        if exact_mask.any():
            exact_indices = self.movies[exact_mask].index
            for idx in exact_indices:
                row = self.movies.loc[idx]
                movie_year = row['year']
                
                score = 100
                if search_year and movie_year and search_year == movie_year:
                    score = 110
                elif search_year and movie_year and search_year != movie_year:
                    score = 95
                    
                candidates.append((idx, row['title'], score, movie_year or 'Unknown'))
        
        # 2. Substring matches - vectorized
        if len(search_normalized) > 0:
            substring_mask = normalized_titles.str.contains(search_normalized, case=False, na=False)
            substring_mask = substring_mask & ~exact_mask  # Exclude already found exact matches
            
            if substring_mask.any():
                substring_indices = self.movies[substring_mask].index
                for idx in substring_indices:
                    row = self.movies.loc[idx]
                    movie_year = row['year']
                    
                    score = 80
                    if search_year and movie_year and search_year == movie_year:
                        score = 90
                        
                    candidates.append((idx, row['title'], score, movie_year or 'Unknown'))
        
        # 3. Reverse substring matches - vectorized  
        reverse_mask = pd.Series(False, index=self.movies.index)
        for idx, normalized_title in normalized_titles.items():
            if pd.notna(normalized_title) and normalized_title and search_normalized in normalized_title:
                if not exact_mask.loc[idx] and (not substring_mask.any() or not substring_mask.loc[idx]):
                    reverse_mask.loc[idx] = True
        
        if reverse_mask.any():
            reverse_indices = self.movies[reverse_mask].index  
            for idx in reverse_indices:
                row = self.movies.loc[idx]
                movie_year = row['year']
                
                score = 70
                if search_year and movie_year and search_year == movie_year:
                    score = 85
                    
                candidates.append((idx, row['title'], score, movie_year or 'Unknown'))
        
        # 4. Word-based fuzzy matching - optimized with set operations
        if len(candidates) < 50:  # Only do expensive fuzzy matching if we don't have enough candidates
            search_words = set(search_normalized.split())
            
            # Use existing fuzzy index for faster lookup
            potential_matches = set()
            for word in search_words:
                if word in self.title_fuzzy_index:
                    potential_matches.update([idx for idx, _ in self.title_fuzzy_index[word]])
            
            # Filter out already found matches
            existing_indices = {candidate[0] for candidate in candidates}
            potential_matches = potential_matches - existing_indices
            
            for idx in potential_matches:
                if idx >= len(self.movies):
                    continue
                    
                row = self.movies.loc[idx]
                movie_normalized = row['normalized_title']
                movie_year = row['year']
                
                if pd.notna(movie_normalized):
                    movie_words = set(movie_normalized.split())
                    
                    if search_words and movie_words:
                        intersection = len(search_words & movie_words)
                        union = len(search_words | movie_words)
                        jaccard = intersection / union if union > 0 else 0
                        
                        if jaccard > 0.6:  # High word overlap
                            score = int(jaccard * 60)
                            if search_year and movie_year and search_year == movie_year:
                                score += 20
                            
                            if score > 30:
                                candidates.append((idx, row['title'], score, movie_year or 'Unknown'))
        
        # Sort by score (descending) and return
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:50]  # Limit results for performance
    def _find_movie_by_title(self, movie_title):
        """Find movie by exact title - expects resolved movie title from MovieSearchUtils"""
        # Check cache first
        if movie_title in self._title_cache:
            return self._title_cache[movie_title]
            
        # Direct exact match (title should be already resolved by MovieSearchUtils)
        exact_matches = self.movies[self.movies['title'] == movie_title]
        if not exact_matches.empty:
            result = exact_matches.index[0]
            self._title_cache[movie_title] = result
            return result
            
        # Fallback: case-insensitive exact match
        case_matches = self.movies[self.movies['title'].str.lower() == movie_title.lower()]
        if not case_matches.empty:
            result = case_matches.index[0]
            self._title_cache[movie_title] = result
            return result
            
        # Not found
        self._title_cache[movie_title] = None
        return None
    
    def search_movies_interactive(self, search_title, max_results=10):
        """
        Interactive movie search with detailed results
        Returns candidates with scores for user to choose from
        """
        candidates = self.find_movie_candidates(search_title)
        
        if not candidates:
            return []
            
        # Format results for display
        results = []
        for idx, title, score, year in candidates[:max_results]:
            results.append({
                'index': idx,
                'title': title,
                'year': year,
                'score': score,
                'movieId': self.movies.iloc[idx]['movieId']
            })
            
        return results
        
    def get_similarity_scores(self, movie_title, genre_weight=0.6, title_weight=0.05, apply_diversity=True):
        """
        Get similarity scores with optional diversity penalty
        Args:
            apply_diversity: If True, applies diversity penalty for better variety
        """
        movie_idx = self._find_movie_by_title(movie_title)
        if movie_idx is None:
            raise ValueError(f"Movie '{movie_title}' not found")
            
        # VECTORIZED similarity calculation
        similarities = np.zeros(len(self.movies), dtype=np.float32)
        
        if genre_weight > 0:
            target_genre_vec = self.genre_tfidf_matrix[movie_idx:movie_idx+1]
            if target_genre_vec.nnz > 0:
                genre_sims = cosine_similarity(target_genre_vec, self.genre_tfidf_matrix).flatten()
                similarities += genre_sims * genre_weight
                
        if title_weight > 0:
            target_title_vec = self.title_tfidf_matrix[movie_idx:movie_idx+1]
            if target_title_vec.nnz > 0:
                title_sims = cosine_similarity(target_title_vec, self.title_tfidf_matrix).flatten()
                similarities += title_sims * title_weight
        
        # Zero out the target movie
        similarities[movie_idx] = 0
        
        # Normalize to [0,1]
        if similarities.max() > 0:
            similarities = similarities / similarities.max()
        
        # Apply diversity penalty if requested
        if apply_diversity:
            similarities = self._apply_diversity_penalty_vectorized(similarities, movie_idx)
            
        return similarities
    
    def _apply_diversity_penalty_vectorized(self, similarities, target_idx):
        """
        FULLY VECTORIZED diversity penalty - optimized for speed
        """
        target_movie = self.movies.iloc[target_idx]
        target_genres = set(target_movie['genres_list'])
        target_year = target_movie['year']
        
        if not target_genres or pd.isna(target_year):
            return similarities
        
        # Vectorized year penalty calculation
        movie_years = self.movies['year'].values
        valid_years_mask = ~pd.isna(movie_years)
        
        if valid_years_mask.any():
            year_diffs = np.abs(movie_years - target_year)
            year_penalties = np.zeros(len(self.movies), dtype=np.float32)
            year_penalties[valid_years_mask & (year_diffs <= 2)] = 0.15
            year_penalties[valid_years_mask & ((year_diffs > 2) & (year_diffs <= 5))] = 0.1
        else:
            year_penalties = np.zeros(len(self.movies), dtype=np.float32)
        
        # Genre penalty calculation (optimized but still needs some iteration)
        genre_penalties = np.zeros(len(self.movies), dtype=np.float32)
        
        # Only calculate for movies with non-zero similarities to save time
        non_zero_indices = np.where((similarities > 0) & (np.arange(len(similarities)) != target_idx))[0]
        
        for idx in non_zero_indices:
            movie_genres = set(self.movies.iloc[idx]['genres_list'])
            if movie_genres:
                intersection = len(target_genres & movie_genres)
                union = len(target_genres | movie_genres)
                genre_overlap = intersection / union if union > 0 else 0
                
                if genre_overlap > 0.8:
                    genre_penalties[idx] = 0.2
        
        # Apply combined penalties
        total_penalties = genre_penalties + year_penalties
        similarities = similarities * (1.0 - total_penalties)
        
        return similarities
    
    @timer
    def recommend_by_movie_fast(self, movie_title, top_k=15, genre_weight=0.6, title_weight=0.05, max_franchise=3):
        """
        STRATEGIC SLOT RECOMMENDATION with narrative priority
        Slots 1-3: Narrative continuation (franchise/sequel) OR high similarity if no franchise
        Slots 4-7: High similarity content 
        Slots 8-10: Same genre | Slots 11+: Diversity
        """
        # Find the target movie
        movie_idx = self._find_movie_by_title(movie_title)
        if movie_idx is None:
            raise ValueError(f"Movie '{movie_title}' not found")
            
        target_movie = self.movies.iloc[movie_idx]
        target_franchise = self._extract_franchise_name(target_movie['title'])
        target_genres = set(target_movie['genres_list'])
        print(f" Strategic recommendations for: {target_movie['title']} (NARRATIVE PRIORITY)")
        
        start_time = time.time()
        
        # Use RAW similarity scores (no diversity penalty for better franchise/similar detection)
        similarities_raw = self.get_similarity_scores(movie_title, genre_weight, title_weight, apply_diversity=False)
        
        calc_time = time.time() - start_time
        print(f" Calculated similarities for {len(self.movies)} movies in {calc_time:.3f} seconds")
        
        # Get large candidate pool for categorization
        candidate_count = min(top_k * 20, len(similarities_raw))
        top_indices = np.argpartition(similarities_raw, -candidate_count)[-candidate_count:]
        top_indices = top_indices[np.argsort(similarities_raw[top_indices])][::-1]
        
        # Categorize candidates by type (using RAW scores)
        narrative_continuation = []  # Franchise/sequel movies
        high_similarity_movies = []
        same_genre_movies = []
        diversity_movies = []
        
        for idx in top_indices:
            if similarities_raw[idx] <= 0:
                continue
                
            movie_row = self.movies.iloc[idx]
            movie_franchise = self._extract_franchise_name(movie_row['title'])
            movie_genres = set(movie_row['genres_list'])
            sim_score = float(similarities_raw[idx])
            movie_data = (movie_row['movieId'], movie_row['title'], sim_score)
            
            # Categorize by priority (using RAW similarity scores)
            if movie_franchise and movie_franchise == target_franchise:
                narrative_continuation.append(movie_data)
            elif sim_score > 0.7:  # Very similar content
                high_similarity_movies.append(movie_data)
            elif target_genres and movie_genres and len(target_genres & movie_genres) > 0:
                same_genre_movies.append(movie_data)
            else:
                diversity_movies.append(movie_data)
        
        # Strategic slot allocation with narrative priority
        selected_movies = []
        
        # Slots 1-3: NARRATIVE CONTINUATION (franchise/sequel)
        narrative_slots = min(3, top_k)
        narrative_count = min(len(narrative_continuation), narrative_slots)
        selected_movies.extend(narrative_continuation[:narrative_count])
        
        # If no narrative continuation available, fill slots 1-3 with high similarity
        remaining_narrative_slots = narrative_slots - narrative_count
        if remaining_narrative_slots > 0:
            selected_movies.extend(high_similarity_movies[:remaining_narrative_slots])
            high_similarity_movies = high_similarity_movies[remaining_narrative_slots:]  # Remove used items
        
        remaining_slots = top_k - len(selected_movies)
        
        # Slots 4-7: High similarity content (remaining)
        if remaining_slots > 0:
            similarity_slots = min(4, remaining_slots)
            selected_movies.extend(high_similarity_movies[:similarity_slots])
            remaining_slots = top_k - len(selected_movies)
        
        # Slots 8-10: Same genre
        if remaining_slots > 0:
            genre_slots = min(3, remaining_slots)
            selected_movies.extend(same_genre_movies[:genre_slots])
            remaining_slots = top_k - len(selected_movies)
        
        # Slots 11+: Diversity (APPLY diversity penalty here)
        if remaining_slots > 0:
            # Apply diversity penalty only to diversity movies
            diversity_with_penalty = []
            if diversity_movies:
                # Get WITH diversity penalty for better diversity
                similarities_with_penalty = self.get_similarity_scores(movie_title, genre_weight, title_weight, apply_diversity=True)
                
                for movie_id, title, raw_score in diversity_movies:
                    # Find movie index to get penalized score (use cached lookup)
                    movie_idx_penalty = self._find_movie_by_title(title)
                    if movie_idx_penalty is not None:
                        penalized_score = float(similarities_with_penalty[movie_idx_penalty])
                        diversity_with_penalty.append((movie_id, title, penalized_score))
                    else:
                        diversity_with_penalty.append((movie_id, title, raw_score))
                
                # Sort by penalized scores for better diversity
                diversity_with_penalty.sort(key=lambda x: x[2], reverse=True)
            
            selected_movies.extend(diversity_with_penalty[:remaining_slots])
        
        # Ensure we have enough results
        if len(selected_movies) < top_k:
            # Fill remaining slots with best available
            all_remaining = narrative_continuation + high_similarity_movies + same_genre_movies + diversity_movies
            used_ids = {movie[0] for movie in selected_movies}
            for movie in all_remaining:
                if movie[0] not in used_ids and len(selected_movies) < top_k:
                    selected_movies.append(movie)
        
        # Updated logging to reflect new strategy
        narrative_filled = min(len(narrative_continuation), 3)
        similarity_filled = min(len(selected_movies), 7) - narrative_filled
        genre_filled = min(len(same_genre_movies), 3) if len(selected_movies) > 7 else 0
        diverse_filled = len(selected_movies) - narrative_filled - similarity_filled - genre_filled
        
        print(f"📊 Slots filled - Narrative: {narrative_filled}, Similarity: {similarity_filled}, Genre: {genre_filled}, Diverse: {diverse_filled}")
        if narrative_filled == 0:
            print("📝 No franchise/sequel found - slots 1-7 filled with high similarity content")
        
        return selected_movies[:top_k]
        
    def recommend_by_genre(self, genres, top_k=15, min_year=None, max_year=None):
        """
        Genre-based filtering for genre-to-movie recommendations
        """
        if isinstance(genres, str):
            target_genres = [g.strip().lower() for g in genres.split('|')]
        else:
            target_genres = [g.lower() for g in genres]
            
        print(f"Finding movies with genres: {', '.join(target_genres)}")
        
        # Filter movies by genre and calculate scores
        genre_matches = []
        
        for idx, row in self.movies.iterrows():
            movie_genres = set(row['genres_list'])
            target_genre_set = set(target_genres)
            
            # Calculate Jaccard similarity for genres
            intersection = len(movie_genres & target_genre_set)
            union = len(movie_genres | target_genre_set)
            
            if intersection > 0:  # Movie has at least one matching genre
                jaccard_score = intersection / union if union > 0 else 0
                
                # Apply year filters if specified
                movie_year = row['year']
                if min_year and movie_year and movie_year < min_year:
                    continue
                if max_year and movie_year and movie_year > max_year:
                    continue
                    
                # Calculate additional score based on genre coverage
                coverage_score = intersection / len(target_genre_set)
                combined_score = (jaccard_score * 0.6) + (coverage_score * 0.4)
                
                # Normalize score to [0,1]
                normalized_score = min(1.0, combined_score)
                
                genre_matches.append((row['movieId'], row['title'], normalized_score))
        
        genre_matches.sort(key=lambda x: x[2], reverse=True)
        return genre_matches[:top_k]
        
    @staticmethod
    def get_search_guidelines():
        """
        Return comprehensive guidelines for movie title search
        """
        guidelines = """
 MOVIE SEARCH GUIDELINES
========================

 RECOMMENDED SEARCH FORMATS:

1. EXACT TITLE MATCHING:
   - "toy story"          → Toy Story (1995)
   - "avengers"           → Avengers, The (2012) 
   - "dark knight"        → Dark Knight, The (2008)

2. WITH YEAR (for specific movies):
   - "avengers 2012"      → Avengers, The (2012)
   - "batman 1989"        → Batman (1989)
   - "spider-man 2002"    → Spider-Man (2002)

3. PARTIAL TITLE MATCHING:
   - "lord rings"         → Lord of the Rings series
   - "star wars"          → Star Wars series
   - "harry potter"       → Harry Potter series

    AVOID THESE FORMATS:
   - "the avengers" (use "avengers" instead)
   - "avengers, the (2012)" (use "avengers 2012")
   - "The Dark Knight" (use "dark knight")
   
    TIPS FOR BETTER RESULTS:
   - Use lowercase letters
   - Skip articles ("the", "a", "an") at the beginning
   - Add year if you want a specific version
   - Use key words from the title
   - Try partial matches if exact doesn't work

    TROUBLESHOOTING:
   - If no results: try shorter keywords
   - If wrong movie: add year to be specific
   - For sequels: use numbers ("toy story 2")
   - For franchises: be specific ("batman begins", "batman returns")
        """
        return guidelines
    
    def print_search_guidelines(self):
        """Print search guidelines to console"""
        print(self.get_search_guidelines())

    def debug_movie_search(self, search_title, show_top=5):
        """
        Debug movie search - shows top candidates with scores
        Useful for understanding why certain searches fail
        """
        print(f"\n DEBUG SEARCH: '{search_title}'")
        print("=" * 50)
        
        candidates = self.find_movie_candidates(search_title)
        
        if not candidates:
            print("    No candidates found!")
            print("\n    Suggestions:")
            print("       - Try shorter keywords")
            print("       - Check spelling")
            print("       - Use partial title match")
            return
        
        print(f" Found {len(candidates)} candidates (showing top {show_top}):")
        print()
        
        for i, (idx, title, score, year) in enumerate(candidates[:show_top]):
            print(f"{i+1}. {title} ({year}) - Score: {score}")
        
        print(f"\n Best match: {candidates[0][1]} ({candidates[0][3]})")
        return candidates
