# Movie Recommendation System

This movie recommendation system helps users find movies they might enjoy by applying multiple recommendation techniques to provide a more personalized movie suggestions. The system integrates collaborative filtering, content-based filtering, and hybrid approaches through a user-friendly web interface.

## Authors

- **@CosmosGoldl**: Nguyễn Anh Thy
- **@xxxibnyn**: Nguyễn Yến Nhi

## Dataset

This project uses the **MovieLens 25M Dataset**, which contains 25 million ratings and one million tag applications applied to 62,000 movies by 162,000 users. The dataset provides comprehensive movie metadata and user interaction data for training robust recommendation algorithms.

## Main Features

- **Collaborative Filtering**: Uses Alternating Least Squares (ALS) algorithm for user-item matrix factorization
- **Content-Based Filtering**: Analyzes movie features like genres, directors, and cast using TF-IDF vectorization
- **Hybrid Recommendation**: Intelligently combines multiple approaches for optimal recommendations
- **Web Interface**: Flask-based web application for easy interaction
- **Enhanced Movie Information**: Integration with OMDB API for additional movie details like posters and plot descriptions
- **Smart Search**: Fuzzy matching and normalization for movie title searches

## Project UI:

<img width="1897" height="927" alt="Screenshot 2026-01-24 103942" src="https://github.com/user-attachments/assets/fe55320a-b211-483d-a84c-401e72a827b0" />
<img width="1898" height="925" alt="Screenshot 2026-01-24 104003" src="https://github.com/user-attachments/assets/416451b2-8623-4305-b813-e19a25d2d03c" />
<img width="1902" height="926" alt="Screenshot 2026-01-24 104101" src="https://github.com/user-attachments/assets/3361f782-0254-4e23-86b4-2ef76f3b4645" />
<img width="1898" height="925" alt="Screenshot 2026-01-24 104527" src="https://github.com/user-attachments/assets/48acdfe9-7d52-45f9-aa8f-958fcc9c4059" />
<img width="1901" height="928" alt="Screenshot 2026-01-24 104640" src="https://github.com/user-attachments/assets/5b533c2f-4b3d-4d95-97c0-11c5bdf3bf74" />
<img width="1901" height="926" alt="Screenshot 2026-01-24 104703" src="https://github.com/user-attachments/assets/4fdffe2f-ee18-491d-bf67-65ea57c40019" />

## Project Structure

```
├── app.py                    # Main Flask application
├── rc_CF.py                  # Collaborative Filtering (ALS) implementation
├── contentbased.py           # Content-based recommendation system
├── hybrid_recommender.py     # Hybrid recommendation combining CF and CB
├── PlsTrain.py               # Model training and data preprocessing
├── als_model.pkl             # Pre-trained ALS model
├── static/
│   ├── css/
│   │   └── style.css         # Frontend styling
│   └── js/
│       └── script.js         # Frontend JavaScript
├── templates/
│   ├── index.html            # Home page
│   ├── recommend.html        # Recommendation results page
│   ├── guidelines.html       # Usage guidelines
│   └── error.html            # Error handling page
├── movies.csv                # Movie metadata (excluded in .gitignore)
├── ratings.csv               # User ratings data (excluded in .gitignore)
├── links.csv                 # Movie ID mappings (excluded in .gitignore)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Implementation Guide

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/CosmosGoldl/Movies-recommendation.git
   cd Movies-recommendation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv movie_rec_env
   source movie_rec_env/bin/activate  # On Windows: movie_rec_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required datasets**
   - Download MovieLens 25M dataset (movies.csv, ratings.csv, links.csv) from https://grouplens.org/datasets/movielens/25m/
   - Place them in the project root directory
   - These files contain 25M ratings, movie metadata, and user interaction data

5. **Train the model (if needed)**
   ```bash
   python PlsTrain.py
   ```
   This will generate the `als_model.pkl` file required for collaborative filtering.

### Configuration

1. **API Key Setup**
   - Get an API key from OMDB (http://www.omdbapi.com/apikey.aspx)
   - Replace the API key in `app.py`:
     ```python
     OMDB_API_KEY = "your_api_key_here"
     ```

2. **Data Path Configuration**
   - Ensure CSV files are in the correct path
   - Modify paths in the code if necessary

### Deployment

#### Local Development
```bash
python app.py
```
The application will be available at `http://localhost:5000`

#### Production Deployment

1. **Using Gunicorn (Linux/Mac)**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Using Waitress (Windows)**
   ```bash
   pip install waitress
   waitress-serve --host=0.0.0.0 --port=5000 app:app
   ```

3. **Docker Deployment**
   Create a Dockerfile:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["python", "app.py"]
   ```

   Build and run:
   ```bash
   docker build -t movie-recommender .
   docker run -p 5000:5000 movie-recommender
   ```

## Usage

1. **Access the web interface** at the configured URL
2. **Enter a movie title** in the search box
3. **Select recommendation type**:
   - Content-Based: Recommends movies similar in features
   - Collaborative Filtering: Recommends based on user behavior patterns
   - Hybrid: Combines both approaches intelligently
4. **View recommendations** with detailed movie information
5. **Explore similar movies** through the interactive interface

## Technical Details

- **Machine Learning**: ALS matrix factorization, TF-IDF vectorization, cosine similarity
- **Backend**: Flask web framework with RESTful API endpoints
- **Frontend**: HTML5, CSS3, JavaScript with responsive design
- **Data Processing**: Pandas for data manipulation, NumPy for numerical computations
- **Model Persistence**: Joblib for saving/loading trained models
- **External APIs**: OMDB API integration for enhanced movie metadata and visual content

## Performance Optimization

- Thread-safe caching for API calls
- Lazy loading of recommendation models
- Efficient sparse matrix operations
- Intelligent hybrid weighting algorithms
- Fuzzy string matching for robust search

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
