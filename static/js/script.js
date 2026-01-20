async function getRecommendations() {
  const userId = document.getElementById("userIdInput").value;
  if (!userId) {
    alert("Vui lòng nhập User ID!");
    return;
  }

  try {
    const response = await fetch(`/api/rc/${userId}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Lỗi khi lấy gợi ý");
    }
    const data = await response.json();
    renderResults(data, `Movies for You`, 'moviesForYouResults', 'moviesForYou');
  } catch (error) {
    alert(error.message);
  }
}

async function getSimilar() {
  const movieName = document.getElementById("movieInput").value.trim();
  if (!movieName) {
    alert("Vui lòng nhập tên phim!");
    return;
  }

  try {
    const response = await fetch(`/api/sim/${encodeURIComponent(movieName)}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Lỗi khi tìm phim tương tự");
    }
    const data = await response.json();
    renderResults(data, `Because You Watched "${movieName}"`, 'becauseYouWatchedResults', 'becauseYouWatched');
  } catch (error) {
    alert(error.message);
  }
}

function renderResults(data, title, sectionId, sectionContainerId) {
  const resultsDiv = document.getElementById(sectionId);
  const sectionContainer = document.getElementById(sectionContainerId);
  const sectionTitle = resultsDiv.parentElement.querySelector('h2');
  sectionTitle.textContent = title;

  resultsDiv.innerHTML = '';

  if (data.error) {
    resultsDiv.innerHTML = `<p style="color:#ff4d4d;">${data.error}</p>`;
    sectionContainer.classList.add('active');
    return;
  }

  for (const movie of data) {
    const poster = movie.poster && movie.poster !== "N/A" ? movie.poster : "https://via.placeholder.com/200x300?text=No+Image";
    const year = movie.year || "";
    const plot = movie.plot || "Không có mô tả.";
    const actors = movie.actors || "Không có thông tin diễn viên.";

    const card = document.createElement("div");
    card.className = "movie-card";
    card.innerHTML = `
      <img src="${poster}" alt="${movie.title}">
      <div class="movie-info">
        <h4>${movie.title}</h4>
        <p>${plot.length > 100 ? plot.substring(0, 100) + "..." : plot}</p>
        <p><b>Score:</b> ${movie.score.toFixed(3)}</p>
        <p><b>Actors:</b> ${actors}</p>
        <a href="#" class="watch-btn">Start Watching</a>
      </div>
    `;
    card.addEventListener('click', () => openModal(movie));
    resultsDiv.appendChild(card);
  }

  // Show the section only if there are results or an error
  sectionContainer.classList.add('active');
}
let cameFromSimilar = false;
function openModal(movie) {
  const modal = document.getElementById('movieModal');
  const modalPoster = document.getElementById('modalPoster');
  const modalTitle = document.getElementById('modalTitle');
  const modalYear = document.getElementById('modalYear');
  const modalPlot = document.getElementById('modalPlot');
  const modalScore = document.getElementById('modalScore');
  const modalImdbRating = document.getElementById('modalImdbRating');
  const modalActors = document.getElementById('modalActors');
 
  const modalPlayBtn2 = document.getElementById('modalPlayBtn2');

  modalPoster.src = movie.poster && movie.poster !== "N/A" ? movie.poster : "https://via.placeholder.com/200x300?text=No+Image";
  modalTitle.textContent = movie.title;
  modalYear.textContent = movie.year || "N/A";
  modalPlot.textContent = movie.plot || "Không có mô tả.";
  modalScore.textContent = movie.score.toFixed(3);
  modalImdbRating.textContent = movie.imdbRating || "N/A";
  modalActors.textContent = movie.actors || "Không có thông tin diễn viên.";

  modalPlayBtn2.href = `#`; // Placeholder link

  modal.style.display = 'block';

  // Close modal when clicking the close button
  document.querySelector('.close-btn').addEventListener('click', () => {
    modal.style.display = 'none';
     if (cameFromSimilar) {
    document.getElementById('similarModal').style.display = 'block';
    cameFromSimilar = false; // reset lại trạng thái
     }
  });

  // Close modal when clicking outside
  window.addEventListener('click', (event) => {
    if (event.target === modal) {
      modal.style.display = 'none';
       if (cameFromSimilar) {
      document.getElementById('similarModal').style.display = 'block';
      cameFromSimilar = false;
    }
    }
  });
}

async function showSimilarMovies(event) {
  event.preventDefault();
  const modalTitle = document.getElementById('modalTitle').textContent.split(' (')[0]; // Extract title without year
  const similarModal = document.getElementById('similarModal');
  const similarResults = document.getElementById('similarResults');

  try {
    const response = await fetch(`/api/sim/${encodeURIComponent(modalTitle)}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Lỗi khi tìm phim tương tự");
    }
    const data = await response.json();
    similarResults.innerHTML = '';

    if (data.error) {
      similarResults.innerHTML = `<p style="color:#ff4d4d;">${data.error}</p>`;
    } else {
      for (const movie of data) {
        const poster = movie.poster && movie.poster !== "N/A" ? movie.poster : "https://via.placeholder.com/200x300?text=No+Image";
        const year = movie.year || "";
        const card = document.createElement("div");
        card.className = "movie-card";
        card.innerHTML = `
          <img src="${poster}" alt="${movie.title}">
          <div class="movie-info">
            <h4>${movie.title}</h4>
            <p><b>Score:</b> ${movie.score.toFixed(3)}</p>
            <p><b>IMDb Rating:</b> ${movie.imdbRating || "N/A"}</p>
          </div>
        `;
        card.addEventListener('click', () => {
          cameFromSimilar = true; // đánh dấu rằng phim này mở ra từ similar list
          document.getElementById('similarModal').style.display = 'none'; // ẩn modal similar
          openModal(movie);
});
        similarResults.appendChild(card);
      }
    }

    similarModal.style.display = 'block';
    similarResults.scrollTo({ top: 0, behavior: "smooth" });

    // Close similar modal when clicking the close button
    document.querySelector('#similarModal .close-btn').addEventListener('click', () => {
      similarModal.style.display = 'none';
    });

    // Close similar modal when clicking outside
    window.addEventListener('click', (event) => {
      if (event.target === similarModal) {
        similarModal.style.display = 'none';
      }
    });
  } catch (error) {
    alert(error.message);
  }
}