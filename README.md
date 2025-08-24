# Spotify Music Data Analysis and Popularity Prediction

A comprehensive data analysis and machine learning project that explores Spotify track features and predicts song popularity using various regression models.

## 📊 Project Overview

This project analyzes a dataset of Spotify tracks to understand musical characteristics and build predictive models for song popularity. The analysis includes exploratory data analysis (EDA), feature engineering, and comparison of multiple machine learning algorithms.

## 🎯 Objectives

- Analyze musical features and their distributions
- Identify top artists and most popular songs
- Explore relationships between audio features and popularity
- Build and compare regression models to predict song popularity
- Determine the most important features for predicting popularity

## 📁 Dataset

The project uses a Spotify tracks dataset (`tracks.csv`) containing the following features:

### Audio Features
- **danceability**: How suitable a track is for dancing (0.0 to 1.0)
- **energy**: Perceptual measure of intensity and power (0.0 to 1.0)
- **valence**: Musical positivity conveyed by a track (0.0 to 1.0)
- **acousticness**: Confidence measure of whether the track is acoustic (0.0 to 1.0)
- **instrumentalness**: Predicts whether a track contains no vocals (0.0 to 1.0)
- **liveness**: Detects presence of an audience in the recording (0.0 to 1.0)
- **speechiness**: Detects presence of spoken words in a track (0.0 to 1.0)
- **tempo**: Overall estimated tempo of a track in BPM
- **loudness**: Overall loudness of a track in decibels (dB)
- **mode**: Modality (major or minor) of a track
- **time_signature**: Time signature of the track

### Metadata
- **popularity**: The popularity score (0-100)
- **duration_ms**: Track length in milliseconds
- **explicit**: Whether the track has explicit lyrics
- **name**: Track name
- **artists**: Artist name(s)
- **release_date**: Release date of the track

## 🛠️ Technologies Used

### Libraries
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Statistical Analysis**: scipy

### Machine Learning Models
- Linear Regression
- Ridge Regression (with hyperparameter tuning)
- Lasso Regression (with hyperparameter tuning)
- Random Forest Regressor (with hyperparameter tuning)

## 🔍 Key Findings

### Data Distribution Analysis
- **Right-skewed features**: popularity, duration, explicit, speechiness, instrumentalness, liveness
- **Left-skewed features**: loudness, mode, time_signature
- **Normal distribution**: danceability, energy, valence, tempo

### Feature Correlations
- Strong positive correlation between energy and loudness (0.64)
- Features selected based on correlation threshold > 0.05 with popularity

### Model Performance
The project compares four regression models using RMSE and R² score metrics:

1. **Linear Regression**
2. **Ridge Regression** (with α tuning: 0.1, 1, 10, 50, 100)
3. **Lasso Regression** (with α tuning: 0.001, 0.01, 0.1, 1, 10)
4. **Random Forest** (with n_estimators, max_depth, min_samples_split tuning)

## 📈 Analysis Highlights

### Top Insights
- Identification of top 20 artists by song count
- Analysis of song release trends over years
- Discovery of most danceable, energetic, and positive songs
- Feature importance ranking using Random Forest

### Visualizations
- Distribution plots for all numerical features
- Correlation heatmap
- Top artists and popular songs bar charts
- Model performance comparison charts
- Feature importance plots

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Running the Analysis
1. Clone the repository
2. Ensure `tracks.csv` is in the correct path
3. Run the Jupyter notebook or Python script:
```bash
python spotify.py
```

### Expected Output
- Exploratory data analysis visualizations
- Model performance metrics
- Feature importance rankings
- Comparative analysis charts

## 📊 Model Evaluation

Models are evaluated using:
- **RMSE (Root Mean Square Error)**: Lower is better
- **R² Score**: Higher is better (closer to 1.0)

The evaluation includes cross-validation and hyperparameter tuning using GridSearchCV for optimal performance.

## 🎵 Use Cases

This analysis can be valuable for:
- **Music Producers**: Understanding features that contribute to popular songs
- **Playlist Curators**: Selecting tracks based on audio characteristics
- **Music Streaming Platforms**: Recommending songs to users
- **Artists**: Analyzing successful track characteristics
- **Music Researchers**: Studying trends in popular music

## 🔮 Future Enhancements

- Implement additional regression models (XGBoost, Neural Networks)
- Add time-series analysis for popularity trends
- Include genre-based analysis
- Implement recommendation system
- Add sentiment analysis of lyrics
- Create interactive dashboard

## 📝 Notes

- The dataset contains outliers in several features (duration_ms, loudness, speechiness, instrumentalness, liveness, tempo)
- Standard scaling is applied for linear models
- Random Forest doesn't require feature scaling
- Feature selection based on correlation threshold helps reduce dimensionality

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or additional analysis techniques.

## 📄 License

This project is available for educational and research purposes.

---

*Happy analyzing! 🎶*
