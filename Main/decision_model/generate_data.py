import random
import pandas as pd

high_value = ["laptop", "phone", "iPad", "bag", "watch", "student-id"]
medium_value = ["earphones", "bottle", "wallet", "keys"]
low_value = ["paper", "pen", "book", "glasses", "mouse", "charging-cable"]

all_items = high_value + medium_value + low_value

def item_value_score(items):
    score = 0
    for it in items:
        if it in high_value:
            score += 2
        elif it in medium_value:
            score += 1
        else:
            score += 0
    return score

def generate_sample():
    items = random.sample(all_items, random.randint(1, 4))
    value_score = item_value_score(items)

    time_since = random.randint(0, 60)
    current_sit = random.randint(0, 200)
    total_session = current_sit + random.randint(0, 200)
    prev_returns = random.randint(0, 5)

    seat_now_occupied = random.choice([0, 1])
    new_person_present = random.choice([0, 1])

    weekday = random.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    time_of_day = round(random.uniform(8, 22), 2)

    # -------- Label Generation Logic --------
    label = None

    # --- OCCUPIED ---
    if time_since == 0:
        label = "OCCUPIED"
    elif seat_now_occupied == 1 and new_person_present == 0:
        label = "OCCUPIED"

    # --- TEMPORARY LEAVE ---
    elif time_since <= 15 and value_score >= 3 and seat_now_occupied == 0:
        label = "TEMP_LEAVE"
    elif time_since <= 10 and prev_returns >= 1:
        label = "TEMP_LEAVE"
    elif value_score >= 4 and time_since <= 20:
        label = "TEMP_LEAVE"

    # --- LEFT BEHIND ---
    elif time_since > 30 and value_score <= 1:
        label = "LEFT_BEHIND"
    elif seat_now_occupied == 1 and new_person_present == 1:
        label = "LEFT_BEHIND"
    elif prev_returns == 0 and time_since > 25:
        label = "LEFT_BEHIND"
    else:
        # fallback logic
        if value_score <= 1:
            label = "LEFT_BEHIND"
        else:
            label = "TEMP_LEAVE"

    return {
        "item_types": ",".join(items),
        "weekday": weekday,
        "time_since_person": time_since,
        "time_of_day": time_of_day,
        "current_sit_minutes": current_sit,
        "total_session_minutes": total_session,
        "num_previous_returns": prev_returns,
        "seat_now_occupied": seat_now_occupied,
        "new_person_present": new_person_present,
        "label": label
    }

# Generate dataset
df = pd.DataFrame([generate_sample() for _ in range(20000)])
df.to_csv("library_lost_item_training_data.csv", index=False)

df.head()
