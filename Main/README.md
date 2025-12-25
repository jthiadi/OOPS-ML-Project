# Object Ownership Project System (OOPS)

This project is an **Object Ownership Protection System** built as part of the *Introduction to Machine Learning* final project.  
It combines several **Machine Learning** components, including **Object Detection**, **Face Recognition**, **Pose Detection**, **Prediction**, combined with **FastAPI backend**, and a **Flutter web frontend**.

The system performs object-related ownership logic processing in the main code and seat state prediction using trained machine learning models. The functionality is exposed through an API that is consumed by a Flutter web application.
## Requirements

- Python 3.11+
- Flutter (with Chrome enabled)
- Internet connection (for model/API usage)

## Configuration

Before running the project, the following values must be configure:
- 'api.py': Set the required API key.
- 'main.py': Configure camera input sources (IP cameras or USB camera indices).

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

## Authors
1. A. M. Wijaya
2. E. L. Suryasatria
3. J. N. Hartono
4. J. Thiadi
5. M. A. Salim
6. S. Indrawan



