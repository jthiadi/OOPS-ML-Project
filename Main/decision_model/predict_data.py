import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lost_item_rf_model.pkl")
model = joblib.load(MODEL_PATH)

def predict_status(
    item_types,
    weekday,
    time_since_person,
    time_of_day,
    current_sit_minutes,
    total_session_minutes,
    num_previous_returns,
    seat_now_occupied,
    new_person_present
):
    """
    Predict LEFT_BEHIND / TEMP_LEAVE / OCCUPIED
    """

    if isinstance(item_types, list):
        item_str = ",".join(sorted(item_types))
    else:
        item_str = ",".join(sorted(item_types.split(",")))

    sample = {
        "item_types": item_str,
        "weekday": weekday,
        "time_since_person": time_since_person,
        "time_of_day": time_of_day,
        "current_sit_minutes": current_sit_minutes,
        "total_session_minutes": total_session_minutes,
        "num_previous_returns": num_previous_returns,
        "seat_now_occupied": seat_now_occupied,
        "new_person_present": new_person_present,
    }

    df = pd.DataFrame([sample])
    ml_label = model.predict(df)[0]                # label
    proba = model.predict_proba(df)[0]             # probabilities
    classes = model.classes_                       # class labels
    
    prob = {cls: float(p) for cls, p in zip(classes, proba)}

    # Confidence = prob of the predicted label
    thr = prob[ml_label]                     

    return prob, ml_label, thr

if __name__ == "__main__":
    print("")
