<div align="center">

# 🏠 California Housing Price Predictor

### *Predict California House Prices with Machine Learning*

  <img src="https://raw.githubusercontent.com/Sanjaymo/california-housing-ml/main/assets/banner.png" alt="California Housing Banner" width="800"/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-ff4b4b.svg" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Model-Linear%20Regression-green.svg" alt="Model"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status"/>
  <img src="https://img.shields.io/github/stars/Sanjaymo/california-housing-ml?style=social" alt="GitHub Stars"/>
  <img src="https://img.shields.io/github/forks/Sanjaymo/california-housing-ml?style=social" alt="GitHub Forks"/>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-contributing">Contributing</a> •
  <a href="#-license">License</a>
</p>

</div>

---

## 📖 About The Project

The **California Housing Price Predictor** is a machine learning application that predicts median house values in California using the famous California Housing dataset. This project demonstrates end-to-end ML workflow including data loading, model training, evaluation, and deployment through both console and web interfaces.

Built with **scikit-learn** for modeling and **Streamlit** for web deployment, this project serves as an excellent starting point for anyone interested in real estate price prediction and interactive ML applications.

---

## ✨ Features

- 🎯 **Accurate Predictions** - Multiple Linear Regression model trained on 20,640 housing samples
- 🖥️ **Dual Interface** - Console-based CLI and interactive Streamlit web app
- 📊 **Visual Analytics** - Actual vs Predicted scatter plots with performance metrics
- 🔄 **Real-time Training** - Retrain the model on-demand with updated parameters
- 🎨 **Interactive UI** - User-friendly sliders and inputs for feature adjustment
- 📈 **Model Metrics** - MSE, R² Score, and MAE for performance evaluation
- 🛡️ **Error Handling** - Robust input validation and error management
- 💾 **Default Values** - Pre-configured example values for quick testing
- 🌐 **Web Deployment Ready** - Easy to deploy on Streamlit Cloud, Heroku, or AWS

---

## 🎬 Demo

### Streamlit Web Application

<div align="center">
  <img src="https://raw.githubusercontent.com/Sanjaymo/california-housing-ml/main/assets/streamlit-demo.gif" alt="Streamlit Demo" width="800"/>
</div>

### Console Application

<div align="center">
  <img src="https://raw.githubusercontent.com/Sanjaymo/california-housing-ml/main/assets/console-demo.png" alt="Console Demo" width="700"/>
</div>

### Model Performance Visualization

<div align="center">
  <img src="https://raw.githubusercontent.com/Sanjaymo/california-housing-ml/main/assets/prediction-graph.png" alt="Prediction Graph" width="700"/>
</div>

---

## 📂 Project Structure

```text
california-housing-ml/
│
├── 📄 housing_predictor.py      # Console app with menu-driven interface
├── 🌐 streamlit_app.py          # Streamlit web application
├── 📋 requirements.txt          # Python dependencies
├── 📝 README.md                 # Project documentation
├── 🙈 .gitignore                # Git ignore file
│
└── 📁 assets/                   # Screenshots and demo files (optional)
    ├── banner.png
    ├── streamlit-demo.gif
    ├── console-demo.png
    └── prediction-graph.png
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sanjaymo/california-housing-ml.git
cd california-housing-ml
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
```text
numpy==1.26.4
pandas==2.2.0
matplotlib==3.8.2
scikit-learn==1.4.0
streamlit==1.31.0
```

---

## 💻 Usage

### 🖥️ Console Application

Launch the menu-driven console application:

```bash
python housing_predictor.py
```

**Menu Options:**
```
╔═══════════════════════════════════════════╗
║   California Housing Price Predictor      ║
╠═══════════════════════════════════════════╣
║  1. 🏡 Predict House Value                ║
║  2. 📊 Show Actual vs Predicted Graph     ║
║  3. 🔄 Retrain Model                      ║
║  4. 🚪 Exit                               ║
╚═══════════════════════════════════════════╝
```

**Input Features:**
- 💰 Median Income (in $10,000s)
- 🏗️ House Age (years)
- 🛏️ Average Rooms per household
- 🚪 Average Bedrooms per household
- 👥 Population in the block
- 👨‍👩‍👧‍👦 Average Occupancy per household
- 🌍 Latitude (geographic coordinate)
- 🗺️ Longitude (geographic coordinate)

### 🌐 Streamlit Web Application

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

**Web App Features:**
- 🎚️ Interactive sliders for all 8 features
- 🔢 Numeric input fields with validation
- 🎯 Real-time prediction on button click
- 💵 Display results in both units ($100,000s) and USD
- 📱 Responsive design for mobile and desktop

<div align="center">
  <img src="https://raw.githubusercontent.com/Sanjaymo/california-housing-ml/main/assets/streamlit-interface.png" alt="Streamlit Interface" width="800"/>
</div>

---

## 📊 Model Details

### Algorithm
- **Model Type:** Multiple Linear Regression
- **Library:** scikit-learn (`LinearRegression`)
- **Training Samples:** 16,512 (80% of dataset)
- **Test Samples:** 4,128 (20% of dataset)

### Dataset
- **Source:** California Housing Dataset (scikit-learn)
- **Total Samples:** 20,640
- **Features:** 8 numeric predictive attributes
- **Target:** Median house value (in $100,000s)

### Feature Descriptions

| Feature | Description | Range |
|---------|-------------|-------|
| `MedInc` | Median income in block | 0.5 - 15.0 |
| `HouseAge` | Median house age in block | 1 - 52 years |
| `AveRooms` | Average number of rooms | 1 - 142 |
| `AveBedrms` | Average number of bedrooms | 0.3 - 35 |
| `Population` | Block population | 3 - 35,682 |
| `AveOccup` | Average house occupancy | 0.7 - 1,243 |
| `Latitude` | Block latitude | 32.5 - 42.0 |
| `Longitude` | Block longitude | -124.3 - -114.3 |

### Performance Metrics

```
📈 Model Evaluation Results:
├─ Mean Squared Error (MSE): ~0.52
├─ R² Score: ~0.60
└─ Mean Absolute Error (MAE): ~0.53
```

---

## 🎯 Example Prediction

### Input:
```python
Median Income: 8.3252
House Age: 41.0
Average Rooms: 6.984
Average Bedrooms: 1.024
Population: 322.0
Average Occupancy: 2.555
Latitude: 37.88
Longitude: -122.23
```

### Output:
```
🏡 Predicted House Value: 4.52 (in $100,000s)
💵 Approximate Value: $452,000
```

---

## 🛠️ Technologies Used

<div align="center">

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core programming language |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Machine learning library |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) | Web application framework |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | Numerical computing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | Data manipulation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white) | Data visualization |

</div>

---

## 🚀 Deployment

### Deploy on Streamlit Cloud (Free & Easy)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `streamlit_app.py` as the main file
5. Click "Deploy"!

### Deploy on Heroku

```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT" > Procfile

# Create setup.sh
echo "mkdir -p ~/.streamlit/
echo '[server]
headless = true
port = $PORT
enableCORS = false
' > ~/.streamlit/config.toml" > setup.sh

# Deploy
heroku create your-app-name
git push heroku main
```

---

## 🎓 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Linear Regression Tutorial](https://scikit-learn.org/stable/modules/linear_model.html)
- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

---

## 🔮 Future Enhancements

- [ ] **Advanced Models**: Random Forest, Gradient Boosting, XGBoost
- [ ] **Hyperparameter Tuning**: Grid Search and Random Search
- [ ] **Feature Engineering**: Create new features from existing ones
- [ ] **Model Comparison**: Side-by-side comparison dashboard
- [ ] **Data Preprocessing**: Scaling, normalization, and outlier handling
- [ ] **Feature Importance**: Visualize which features matter most
- [ ] **Model Persistence**: Save/load models with joblib
- [ ] **API Endpoint**: REST API for predictions
- [ ] **Docker Support**: Containerize the application
- [ ] **Unit Tests**: Comprehensive test coverage
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Interactive Maps**: Geographic visualization of predictions

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**!

### How to Contribute

1. **Fork the Project**
   ```bash
   # Click the 'Fork' button at the top right of this page
   ```

2. **Create your Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open a Pull Request**
   - Go to your forked repository
   - Click "Compare & pull request"
   - Submit your PR with a clear description

### Contribution Guidelines

- ✅ Follow PEP 8 style guidelines
- ✅ Write clear commit messages
- ✅ Add tests for new features
- ✅ Update documentation as needed
- ✅ Be respectful and constructive

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` file for more information.

```
MIT License

Copyright (c) 2024 Sanjay Choudhari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👨‍💻 Author

<div align="center">

### Sanjay Choudhari

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Sanjaymo)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sanjaychoudhari09/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sanjaychoudhari288@gmail.com)

</div>

---

## 🙏 Acknowledgments

- **Dataset:** [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) from scikit-learn
- **Inspiration:** Kaggle community and open-source ML projects
- **Tools:** Built with Python, scikit-learn, Streamlit, and lots of ☕

---

## 📞 Support

If you have any questions or need help with the project:

- 📧 Email:sanjaychoudhari288@gmail.com
- 💬 Open an [Issue](https://github.com/sanjaychoudhari/california-housing-ml/issues)
- 📖 Check the [Documentation](https://github.com/Sanjaymo/california-housing-ml/wiki)

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

<div align="center">

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sanjaychoudhari/california-housing-ml&type=Date)](https://star-history.com/#sanjaychoudhari/california-housing-ml&Date)

</div>

---

<div align="center">

**Made with ❤️ and Python**

*Version 1.0.0 - December 2025*

[⬆ Back to Top](#-california-housing-price-predictor)

</div>
