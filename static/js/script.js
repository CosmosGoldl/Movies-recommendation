/**
 * GLOBAL STATE VARIABLES
 */
let selectedMovies = new Set();
let cameFromSimilar = false;

/**
 * --- GUIDELINES MODAL FUNCTIONS ---
 */
function openGuidelinesModal() {
    const modal = document.getElementById('guidelinesModal');
    modal.style.display = 'block';
}

function closeGuidelinesModal() {
    const modal = document.getElementById('guidelinesModal');
    modal.style.display = 'none';
}

/**
 * --- PART 1: ONBOARDING LOGIC (SELECT MOVIES FOR PERSONALIZATION) ---
 */

async function startOnboarding() {
    const modal = document.getElementById('onboardingModal');
    selectedMovies.clear();
    updateSelectionUI();
    modal.style.display = 'block';
    
    await loadSampleMovies();
}

async function loadSampleMovies(refresh = false) {
    const grid = document.getElementById('onboardingGrid');
    
    // Improved loading state
    grid.innerHTML = `
        <div style="text-align:center; width:100%; padding:20px;">
            <div style="display:inline-block; width:40px; height:40px; border:3px solid #333; border-top:3px solid #E50914; border-radius:50%; animation:spin 1s linear infinite;"></div>
            <p style="color:#ccc; margin-top:15px;">Loading popular movies...</p>
            <p style="color:#888; font-size:0.9em;">Please wait while we fetch movie details and posters</p>
        </div>
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    `;

    try {
        const url = refresh ? '/api/movies/sample?refresh=true' : '/api/movies/sample';
        const response = await fetch(url);
        if (!response.ok) throw new Error("Failed to load sample movie list");
        
        const data = await response.json();
        
        // Handle new API response structure
        const movies = data.movies || data; // Backward compatibility
        const metadata = data.metadata;
        
        if (!movies || movies.length === 0) {
            throw new Error("No movies available for selection");
        }
        
        grid.innerHTML = '';
        
        // Show metadata if available
        if (metadata) {
            const infoDiv = document.createElement('div');
            infoDiv.style.cssText = 'grid-column: 1 / -1; text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 6px; margin-bottom: 15px;';
            infoDiv.innerHTML = `
                <p style="color: #4CAF50; margin: 0; font-size: 0.9em;">
                     ${movies.length} curated movies loaded • ${Math.round(metadata.modern_ratio * 100)}% modern films
                </p>
            `;
            grid.appendChild(infoDiv);
        }

        movies.forEach((movie, index) => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            card.style.minWidth = 'auto';
            card.style.cursor = 'pointer';
            card.style.transition = 'transform 0.2s, box-shadow 0.2s';

            const posterUrl = movie.poster && movie.poster !== "N/A" 
                            ? movie.poster 
                            : `https://via.placeholder.com/150x225/333/666?text=No+Image`;

            card.innerHTML = `
                <img src="${posterUrl}" alt="${movie.title}" 
                     style="width:100%; border-radius:4px; transition: filter 0.3s;" 
                     onload="this.style.opacity=1" 
                     style="opacity:0">
                <div style="font-size: 0.7em; padding: 5px; text-align: center; color: #ccc; line-height:1.2;">
                    ${movie.title}
                </div>
            `;

            // Add hover effect
            card.onmouseenter = () => {
                if (!selectedMovies.has(movie.movieId)) {
                    card.style.transform = 'scale(1.05)';
                    card.style.boxShadow = '0 4px 15px rgba(229, 9, 20, 0.3)';
                }
            };
            card.onmouseleave = () => {
                if (!selectedMovies.has(movie.movieId)) {
                    card.style.transform = 'scale(1)';
                    card.style.boxShadow = 'none';
                }
            };

            card.onclick = () => toggleMovieSelection(movie.movieId, card);
            grid.appendChild(card);
        });
        
        // Show instruction after loading
        const instruction = document.createElement('div');
        instruction.style.cssText = 'grid-column: 1 / -1; text-align: center; padding: 15px; background: rgba(229, 9, 20, 0.1); border-radius: 8px; margin-top: 10px;';
        instruction.innerHTML = '<p style="color: #E50914; margin: 0;"> Click on movies you like to select them</p><p style="color: #999; font-size: 0.9em; margin: 5px 0 0 0;">Don\'t see movies you like? Use "Load Different Movies" button</p>';
        grid.appendChild(instruction);
        
    } catch (error) {
        grid.innerHTML = `
            <div style="text-align:center; width:100%; padding:20px;">
                <p style="color:#ff4d4d; font-size:1.1em;">${error.message}</p>
                <button onclick="startOnboarding()" style="background:#E50914; color:white; border:none; padding:10px 20px; border-radius:5px; cursor:pointer; margin-top:10px;">Try Again</button>
            </div>
        `;
    }
}

async function reloadSampleMovies() {
    // Reset selected movies
    selectedMovies.clear();
    updateSelectionUI();
    
    // Call existing loadSampleMovies with refresh=true
    await loadSampleMovies(true);
}

function toggleMovieSelection(movieId, element) {
    if (selectedMovies.has(movieId)) {
        // Deselect movie
        selectedMovies.delete(movieId);
        element.style.outline = "none";
        element.style.transform = "scale(1)";
        element.style.boxShadow = "none";
        element.querySelector('img').style.filter = "none";
        element.querySelector('img').style.opacity = "1";
        
        // Remove selection indicator
        const indicator = element.querySelector('.selection-indicator');
        if (indicator) indicator.remove();
    } else {
        // Select movie
        selectedMovies.add(movieId);
        element.style.outline = "3px solid #E50914";
        element.style.transform = "scale(0.95)";
        element.style.boxShadow = "0 0 20px rgba(229, 9, 20, 0.7)";
        element.querySelector('img').style.filter = "brightness(0.7)";
        element.querySelector('img').style.opacity = "0.8";
        
        // Add checkmark indicator
        const indicator = document.createElement('div');
        indicator.className = 'selection-indicator';
        indicator.innerHTML = '✓';
        indicator.style.cssText = `
            position: absolute;
            top: 5px;
            right: 5px;
            background: #E50914;
            color: white;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        `;
        element.style.position = 'relative';
        element.appendChild(indicator);
    }
    updateSelectionUI();
}

function updateSelectionUI() {
    const count = selectedMovies.size;
    const countLabel = document.getElementById('selectionCount');
    const submitBtn = document.getElementById('submitOnboarding');

    if (countLabel) {
        if (count === 0) {
            countLabel.innerHTML = `<span style="color: #ccc;">Selected: ${count} movies</span> - <span style="color: #E50914;">Choose at least 3 movies you enjoy</span>`;
        } else if (count < 3) {
            countLabel.innerHTML = `<span style="color: #ffa500;">Selected: ${count} movies</span> - <span style="color: #E50914;">Need ${3 - count} more to continue</span>`;
        } else {
            countLabel.innerHTML = `<span style="color: #4CAF50;">✓ Selected: ${count} movies</span> - <span style="color: #4CAF50;">Ready to get recommendations!</span>`;
        }
    }

    if (submitBtn) {
        if (count >= 3) {
            submitBtn.disabled = false;
            submitBtn.style.opacity = "1";
            submitBtn.style.cursor = "pointer";
            submitBtn.style.backgroundColor = "#E50914";
            submitBtn.textContent = `🎬 Get My Recommendations (${count} movies selected)`;
        } else {
            submitBtn.disabled = true;
            submitBtn.style.opacity = "0.5";
            submitBtn.style.cursor = "not-allowed";
            submitBtn.style.backgroundColor = "#666";
            if (count === 0) {
                submitBtn.textContent = "Select 3 movies to start";
            } else {
                submitBtn.textContent = `Select ${3 - count} more movie${3 - count > 1 ? 's' : ''}`;
            }
        }
    }
}

async function submitSelection() {
    if (selectedMovies.size < 3) {
        alert(`Please select at least 3 movies. You have selected ${selectedMovies.size} movie${selectedMovies.size !== 1 ? 's' : ''}.`);
        return;
    }
    
    const submitBtn = document.getElementById('submitOnboarding');
    const originalText = submitBtn.textContent;
    
    try {
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.style.opacity = "0.7";
        submitBtn.innerHTML = `
            <span style="display:inline-block; width:15px; height:15px; border:2px solid #fff; border-top:2px solid transparent; border-radius:50%; animation:spin 0.8s linear infinite; margin-right:8px;"></span>
            Analyzing your preferences...
        `;
        
        const response = await fetch('/api/recommend_personal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movieIds: Array.from(selectedMovies) })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Error processing personal recommendations");
        }

        const data = await response.json();
        closeOnboarding();
        
        // 1. Render System Recommend results (Adaptive Hybrid)
        if (data.system_results && data.system_results.length > 0) {
            renderSystemResults(
                data.system_results, 
                `Personalized Recommendations (${selectedMovies.size} movies analyzed)`, 
                'systemRecommendResults', 
                'systemRecommend'
            );
        }
        
        // 2. Render Collaborative Filtering results
        if (data.cf_results && data.cf_results.length > 0) {
            renderResults(
                data.cf_results, 
                "Users Like You Also Enjoyed", 
                'moviesForYouResults', 
                'moviesForYou'
            );
        }
        
        // 3. Render Content-Based results
        if (data.cb_results && data.cb_results.length > 0) {
            renderResults(
                data.cb_results, 
                "Similar Movies Based on Your Taste", 
                'contentBasedResults', 
                'contentBased'
            );
        }

        // Scroll to results with smooth animation
        setTimeout(() => {
            const systemSection = document.getElementById('systemRecommend');
            if (systemSection && systemSection.classList.contains('active')) {
                systemSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                document.getElementById('moviesForYou').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 300);
        
    } catch (error) {
        console.error('Recommendation error:', error);
        alert("❌ " + error.message);
        
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.style.opacity = "1";
        submitBtn.textContent = originalText;
    }
}

function closeOnboarding() {
    document.getElementById('onboardingModal').style.display = 'none';
}

/**
 * --- PART 2: RECOMMENDATION & SEARCH FUNCTIONS ---
 */

async function getRecommendations() {
    const userId = document.getElementById("userIdInput").value;
    if (!userId) {
        alert("Please enter a User ID!");
        return;
    }

    try {
        const response = await fetch(`/api/rc/${userId}`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error || "Error fetching recommendations");
        
        renderResults(data, `Movies for User ${userId}`, 'moviesForYouResults', 'moviesForYou');
    } catch (error) {
        alert(error.message);
    }
}

async function getSimilar() {
    const movieName = document.getElementById("movieInput").value.trim();
    if (!movieName) {
        alert("Please enter a movie title!");
        return;
    }

    try {
        const response = await fetch(`/api/sim/${encodeURIComponent(movieName)}`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error || "Movie not found");
        
        renderResults(data, `Because You Watched "${movieName}"`, 'becauseYouWatchedResults', 'becauseYouWatched');
    } catch (error) {
        alert(error.message);
    }
}

async function getContentBased() {
    const movieName = document.getElementById("contentMovieInput").value.trim();
    if (!movieName) {
        alert("Please enter a movie title for content search!");
        return;
    }

    try {
        const response = await fetch(`/api/content/${encodeURIComponent(movieName)}`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error || "Content-based search failed");
        
        renderResults(data, `Content Similar to "${movieName}"`, 'contentBasedResults', 'contentBased');
    } catch (error) {
        alert(error.message);
    }
}

async function getGenreBased() {
    const genres = document.getElementById("genreInput").value.trim();
    if (!genres) {
        alert("Please enter a genre (e.g., Action|Comedy)!");
        return;
    }

    try {
        const response = await fetch(`/api/genre/${encodeURIComponent(genres)}`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error || "Genre recommendation failed");
        
        renderResults(data, `Top ${genres} Movies`, 'genreBasedResults', 'genreBased');
    } catch (error) {
        alert(error.message);
    }
}

/**
 * --- PART 3: UI RENDERING AND MODALS ---
 */

function renderSystemResults(data, title, sectionId, sectionContainerId) {
    const resultsDiv = document.getElementById(sectionId);
    const sectionContainer = document.getElementById(sectionContainerId);
    if (!resultsDiv || !sectionContainer) return;

    const sectionTitle = sectionContainer.querySelector('h2');
    if (sectionTitle) sectionTitle.textContent = title;

    resultsDiv.innerHTML = '';

    if (!data || data.length === 0) {
        resultsDiv.innerHTML = `<p style="color:#ccc;">No results found.</p>`;
    } else {
        data.forEach((movie, index) => {
            const poster = movie.poster && movie.poster !== "N/A" ? movie.poster : "https://via.placeholder.com/200x300?text=No+Image";
            const plot = movie.plot || "No description available.";
            
            // Hiện thị thông tin hybrid method, confidence và weights
            let hybridInfo = '';
            if (movie.hybrid_method && movie.cf_confidence) {
                hybridInfo = `<p style="font-size:0.7em;color:#E50914;margin:2px 0;"><b>🤖 Method:</b> ${movie.hybrid_method.replace('_', ' ').toUpperCase()}</p>`;
                hybridInfo += `<p style="font-size:0.7em;color:#888;margin:2px 0;"><b>CF Confidence:</b> ${movie.cf_confidence}</p>`;
                
                if (index === 0 && movie.cf_weight && movie.content_weight) {
                    hybridInfo += `<p style="font-size:0.7em;color:#888;margin:2px 0;"><b>Weights:</b> CF=${movie.cf_weight} | CB=${movie.content_weight}</p>`;
                }
            }

            const card = document.createElement("div");
            card.className = "movie-card";
            // Removed red border for cleaner look
            card.innerHTML = `
                <img src="${poster}" alt="${movie.title}">
                <div class="movie-info">
                    <h4>${movie.title}</h4>
                    <p>${plot.length > 60 ? plot.substring(0, 60) + "..." : plot}</p>
                    <p><b>Match Score:</b> ${movie.score ? movie.score.toFixed(3) : "N/A"}</p>
                    ${hybridInfo}
                    <a href="#" class="watch-btn">Start Watching</a>
                </div>
            `;
            card.addEventListener('click', () => openModal(movie));
            resultsDiv.appendChild(card);
        });
    }
    sectionContainer.classList.add('active');
}

function renderResults(data, title, sectionId, sectionContainerId) {
    const resultsDiv = document.getElementById(sectionId);
    const sectionContainer = document.getElementById(sectionContainerId);
    if (!resultsDiv || !sectionContainer) return;

    const sectionTitle = sectionContainer.querySelector('h2');
    if (sectionTitle) sectionTitle.textContent = title;

    resultsDiv.innerHTML = '';

    if (!data || data.length === 0) {
        resultsDiv.innerHTML = `<p style="color:#ccc;">No results found.</p>`;
    } else {
        data.forEach(movie => {
            const poster = movie.poster && movie.poster !== "N/A" ? movie.poster : "https://via.placeholder.com/200x300?text=No+Image";
            const plot = movie.plot || "No description available.";

            const card = document.createElement("div");
            card.className = "movie-card";
            card.innerHTML = `
                <img src="${poster}" alt="${movie.title}">
                <div class="movie-info">
                    <h4>${movie.title}</h4>
                    <p>${plot.length > 80 ? plot.substring(0, 80) + "..." : plot}</p>
                    <p><b>Match Score:</b> ${movie.score ? movie.score.toFixed(3) : "N/A"}</p>
                    <a href="#" class="watch-btn">Start Watching</a>
                </div>
            `;
            card.addEventListener('click', () => openModal(movie));
            resultsDiv.appendChild(card);
        });
    }
    sectionContainer.classList.add('active');
}

function openModal(movie) {
    const modal = document.getElementById('movieModal');
    
    document.getElementById('modalPoster').src = movie.poster && movie.poster !== "N/A" ? movie.poster : "https://via.placeholder.com/200x300?text=No+Image";
    document.getElementById('modalTitle').textContent = movie.title;
    document.getElementById('modalYear').textContent = movie.year || "N/A";
    document.getElementById('modalPlot').textContent = movie.plot || "No description available.";
    document.getElementById('modalScore').textContent = movie.score ? movie.score.toFixed(3) : "N/A";
    document.getElementById('modalImdbRating').textContent = movie.imdbRating || "N/A";
    document.getElementById('modalActors').textContent = movie.actors || "N/A";

    modal.style.display = 'block';

    document.querySelector('.close-btn').onclick = () => {
        modal.style.display = 'none';
        if (cameFromSimilar) {
            document.getElementById('similarModal').style.display = 'block';
            cameFromSimilar = false;
        }
    };
}

async function showSimilarMovies(event) {
    if (event) event.preventDefault();
    const currentTitle = document.getElementById('modalTitle').textContent;
    const similarModal = document.getElementById('similarModal');
    const similarResults = document.getElementById('similarResults');

    try {
        // Sử dụng adaptive hybrid thay vì CF thuần
        const response = await fetch(`/api/adaptive/${encodeURIComponent(currentTitle)}`);
        if (!response.ok) throw new Error("Error finding similar movies");
        
        const data = await response.json();
        similarResults.innerHTML = '';

        data.forEach(movie => {
            const poster = movie.poster && movie.poster !== "N/A" ? movie.poster : "https://via.placeholder.com/200x300?text=No+Image";
            const card = document.createElement("div");
            card.className = "movie-card";
            
            // Hiển thị thông tin hybrid method, confidence và weights nếu có
            let hybridInfo = '';
            if (movie.hybrid_method && movie.cf_confidence) {
                hybridInfo = `<p style="font-size:0.7em;color:#888;"><b>Method:</b> ${movie.hybrid_method} <b>CF:</b> ${movie.cf_confidence}`;
                if (movie.cf_weight && movie.content_weight) {
                    hybridInfo += ` <b>Weights:</b> CF=${movie.cf_weight} CB=${movie.content_weight}`;
                }
                hybridInfo += '</p>';
            }
            
            card.innerHTML = `
                <img src="${poster}" alt="${movie.title}">
                <div class="movie-info">
                    <h4>${movie.title}</h4>
                    <p><b>Score:</b> ${movie.score ? movie.score.toFixed(3) : "N/A"}</p>
                    ${hybridInfo}
                </div>
            `;
            card.onclick = () => {
                cameFromSimilar = true;
                similarModal.style.display = 'none';
                openModal(movie);
            };
            similarResults.appendChild(card);
        });
        similarModal.style.display = 'block';
    } catch (error) {
        alert(error.message);
    }
}

/**
 * --- PART 4: CLICK EVENTS ---
 */

function closeSimilarModal() {
    document.getElementById('similarModal').style.display = 'none';
}

window.addEventListener('click', (event) => {
    const movieModal = document.getElementById('movieModal');
    const similarModal = document.getElementById('similarModal');
    const onboardingModal = document.getElementById('onboardingModal');
    const guidelinesModal = document.getElementById('guidelinesModal');

    if (event.target === movieModal) {
        movieModal.style.display = 'none';
    } else if (event.target === similarModal) {
        similarModal.style.display = 'none';
    } else if (event.target === onboardingModal) {
        onboardingModal.style.display = 'none';
    } else if (event.target === guidelinesModal) {
        guidelinesModal.style.display = 'none';
    }
});