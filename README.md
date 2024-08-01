# Location Prediction App
#Dragon04
https://dragon04.streamlit.app/
## Project Description

The Location Prediction App is a web application built with Streamlit and Folium that visualizes geographic coordinates and provides predictive modeling for location-based data. The app allows users to click on a world map to select a location, and then uses a machine learning model to predict and display nearby locations based on the selected coordinates.

## Features

- Interactive map rendering using Folium.
- Clickable map to select a location.
- Predictive modeling using LightGBM for latitude and longitude predictions.
- Visualization of model outputs on the map.

## Installation

To run this project locally, you'll need to set up your environment and install the necessary dependencies. Follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/location-prediction-app.git
    cd location-prediction-app
    ```

2. **Set up a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare your data:**
   
   Ensure that you have the `new.csv` data file in the same directory as the script. This file should contain latitude and longitude columns along with other features used for prediction.

## Usage

1. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

2. **Access the app:**

   Open your web browser and navigate to `local host`.

3. **Interact with the app:**
   - Click on the map to select a location.
   - The app will display the coordinates of the clicked location.
   - The model will predict nearby locations and update the map with the predicted points.

## Code Structure

- `app.py`: Main Streamlit application script.
- `new.csv`: Data file used for training the machine learning model.
- `requirements.txt`: List of Python packages required for the project.
- `model_latitude.txt` and `model_longitude.txt`: Saved LightGBM models for latitude and longitude predictions.

## Requirements

The project requires the following Python packages:

- `pandas`
- `numpy`
- `math`
- `lightgbm`
- `scikit-learn`
- `streamlit`
- `folium`
- `streamlit_folium`

These packages are listed in the `requirements.txt` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) - For providing a powerful tool to build interactive web applications.
- [Folium](https://python-visualization.github.io/folium/) - For creating interactive maps.
- [LightGBM](https://lightgbm.readthedocs.io/) - For the gradient boosting framework used for predictive modeling.

## Contact

For any questions or issues, please open an issue on this repository or contact me at [your-email@example.com].

