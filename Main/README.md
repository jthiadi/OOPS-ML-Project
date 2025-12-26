## Requirements

- Python 3.11+
- Flutter (with Chrome enabled)
- Internet connection (for model/API usage)

## Configuration

Before running the project, the following values must be configure:
- 'Main/api.py': Set the required API keys.
- 'Main/main.py': Configure camera input sources (IP cameras or USB camera indices).

# Project Inferance
```bash
cd Main/
```
## Setup Instructions

1. Create virtual environtment
```bash
python -m venv venv
```
2. Activate virtual environtment
```bash
venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

1. Start the FastAPI backend
```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
2. Run the Flutter web frontend
```bash
flutter run -d chrome
```
3. Run the main script
```bash
python main.py
```

# Left-Behind Model 
```bash
cd decision_model
```

1. Synthetic data generation
```bash
python generate_data.py
```
2. Prediction model training
```bash
python train.py
```

The model will be located in Main/decision_model/ as lost_item_rf_model.pkl




