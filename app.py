import csv
import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import json
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Mock database (CSV file for persistent storage)
users_file = "fitness_users.csv"
exercise_logs_file = "exercise_logs.csv"
# Load or initialize users' data
try:
    users_data = pd.read_csv(users_file)
except FileNotFoundError:
    users_data = pd.DataFrame(columns=["Name", "Age", "Gender", "Weight", "Height", "BMI", "ActivityLevel"])


if not os.path.exists(exercise_logs_file):
    with open(exercise_logs_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "ExerciseType", "Duration", "WeightBefore", "WeightAfter"])


# Helper function to calculate BMI
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return round(weight / (height_m ** 2), 2)

def calculate_intensity(duration):
    # Example calculation: duration in minutes represents intensity directly
    return duration  # Adjust this as needed to scale the intensity

def generate_exercise_intensity_graphs(user_logs):
    # Convert the logs into a DataFrame
    df = pd.DataFrame(user_logs, columns=["Name", "Date", "Exercise_Type", "Duration", "Weight_Before", "Weight_After"])
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Coerce errors to NaT if invalid
    
    # Convert 'Duration' column to numeric
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')  # Coerce errors to NaN if invalid

    # Drop rows where 'Date' or 'Duration' could not be converted
    df = df.dropna(subset=['Date', 'Duration'])

    # List of unique exercise types
    exercise_types = df['Exercise_Type'].unique()
    graphs = {}

    for exercise in exercise_types:
        # Filter data for the current exercise type
        exercise_data = df[df['Exercise_Type'] == exercise]
        
        # Calculate intensity based on the duration
        exercise_data['Intensity'] = exercise_data['Duration'].apply(calculate_intensity)
        
        # Calculate changes in intensity between consecutive entries
        exercise_data['Intensity Change'] = exercise_data['Intensity'].diff()
        
        # Create a graph for the current exercise type
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=exercise_data['Date'],
            y=exercise_data['Intensity'],
            mode='markers+lines',
            name=f'{exercise} Intensity',
            line=dict(color='blue')
        ))
        
        # Add a line showing intensity changes
        fig.add_trace(go.Scatter(
            x=exercise_data['Date'],
            y=exercise_data['Intensity Change'],
            mode='markers+lines',
            name='Change in Intensity',
            line=dict(color='red', dash='dot')
        ))

        # Customize the layout
        fig.update_layout(
            title=f'Intensity Over Time for {exercise}',
            xaxis_title='Date',
            yaxis_title='Intensity (Duration)',
            template='plotly_dark'
        )
        
        # Save graph as JSON for rendering in the template
        graphs[exercise] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphs

def calculate_predictions_based_on_logs(user_logs, initial_weight, target_weight):
    if len(user_logs) < 2:
        return None, None, None  # Not enough data for predictions

    # Create a DataFrame from the logs
    df = pd.DataFrame(user_logs, columns=["Name", "Date", "Exercise", "Duration", "Weight_Before", "Weight_After"])
    df["Date"] = pd.to_datetime(df["Date"])  # Convert dates to datetime
    df["Weight_After"] = df["Weight_After"].astype(float)

    # Sort logs by date
    df = df.sort_values("Date")
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days

    # Use linear regression to predict weight
    X = df[["Days"]]
    y = df["Weight_After"]
    model = LinearRegression().fit(X, y)

    # Predict weights for the next 7 days
    future_days = np.arange(df["Days"].max() + 1, df["Days"].max() + 8).reshape(-1, 1)
    predictions = model.predict(future_days)

    # Calculate the number of weeks required to reach normal BMI
    rate_of_change_per_day = model.coef_[0]  # Change in weight per day
    if rate_of_change_per_day != 0:
        days_to_normal_bmi = abs((target_weight - df["Weight_After"].iloc[-1]) / rate_of_change_per_day)
        weeks_to_normal_bmi = np.ceil(days_to_normal_bmi / 7)
    else:
        weeks_to_normal_bmi = None  # No weight change predicted

    return future_days.flatten().tolist(), predictions.tolist(), weeks_to_normal_bmi

def predict_weights_from_csv(user_name):
    # Load historical data from CSV
    logs = pd.read_csv("exercise_logs.csv")  # Replace with actual path
    user_logs = logs[logs["Name"] == user_name]

    # Ensure there is enough data for regression
    if len(user_logs) < 2:
        return None, None  # Not enough data for prediction

    # Extract weeks, weights, and workout durations
    weeks = np.arange(1, len(user_logs) + 1).reshape(-1, 1)
    weights = user_logs["Weight"].values.reshape(-1, 1)
    workouts = user_logs["Workout_Duration"].values.reshape(-1, 1)  # Example column name

    # Combine features for multivariate regression
    X = np.hstack((weeks, workouts))

    # Fit the Linear Regression Model
    model = LinearRegression()
    model.fit(X, weights)

    # Predict future weights for the next N weeks
    future_weeks = np.arange(len(weeks) + 1, len(weeks) + 13).reshape(-1, 1)  # Next 12 weeks
    future_workouts = np.full((12, 1), workouts[-1])  # Assume last workout duration repeats
    future_X = np.hstack((future_weeks, future_workouts))

    predicted_weights = model.predict(future_X).flatten()

    return weeks.flatten(), weights.flatten(), future_weeks.flatten(), predicted_weights

def calculate_predictions(user_logs, initial_weight):
    if len(user_logs) < 2:
        return None, None  # Not enough data for predictions

    # Create a DataFrame from the logs
    df = pd.DataFrame(user_logs, columns=["Name", "Date", "Exercise", "Duration", "Weight_Before", "Weight_After"])
    df["Date"] = pd.to_datetime(df["Date"])  # Convert dates to datetime
    df["Weight_After"] = df["Weight_After"].astype(float)

    # Calculate the difference in days from the first log
    df = df.sort_values("Date")
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days

    # Linear regression to predict weight
    X = df[["Days"]]
    y = df["Weight_After"]
    model = LinearRegression().fit(X, y)

    # Predict weights for the next 7 days
    future_days = np.arange(df["Days"].max() + 1, df["Days"].max() + 8).reshape(-1, 1)
    predictions = model.predict(future_days)

    return future_days.flatten().tolist(), predictions.tolist()

def calculate_weight_reduction_weeks(user_logs, current_weight, target_weight):
    """
    Calculates the predicted weeks required to reduce weight to target using linear regression.

    Parameters:
        user_logs (list): Exercise logs including weight and duration.
        current_weight (float): Current weight of the user.
        target_weight (float): Target weight based on BMI.

    Returns:
        days (list): Days for future prediction.
        predicted_weights (list): Predicted weights for those days.
        weeks_required (int): Number of weeks required to reach the target weight.
    """
    if len(user_logs) < 2:
        return None, None, None  # Not enough data for predictions

    # Create DataFrame from logs
    df = pd.DataFrame(user_logs, columns=["Name", "Date", "ExerciseType", "Duration", "WeightBefore", "WeightAfter"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["WeightAfter"] = pd.to_numeric(df["WeightAfter"], errors="coerce")
    df.dropna(subset=["WeightAfter"], inplace=True)

    # Add a column for days from the start
    df.sort_values("Date", inplace=True)
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days

    # Fit a linear regression model
    X = df[["Days"]]
    y = df["WeightAfter"]
    model = LinearRegression().fit(X, y)

    # Predict future weights
    days_future = np.arange(df["Days"].max() + 1, df["Days"].max() + 50).reshape(-1, 1)
    predicted_weights = model.predict(days_future)

    # Calculate the number of weeks to reach the target weight
    rate_of_change_per_day = model.coef_[0]
    if rate_of_change_per_day != 0:
        days_to_target = abs((target_weight - current_weight) / rate_of_change_per_day)
        weeks_required = np.ceil(days_to_target / 7)
    else:
        weeks_required = None  # No change predicted

    return days_future.flatten().tolist(), predicted_weights.tolist(), weeks_required


@app.route("/")
def home():
    # List all users
    return render_template("home.html", users=users_data.to_dict(orient="records"))

@app.route("/user/<name>")
def user_dashboard(name):
    # Get user data
    user = users_data[users_data["Name"] == name].iloc[0]
    user_weight = [user["Weight"]]  # Historical weight data

    # Calculate the normal BMI weight range
    normal_bmi_lower = 18.5
    normal_bmi_upper = 24.9

    # Calculate target weight range based on normal BMI range
    height_m = user["Height"] / 100
    target_weight_lower = normal_bmi_lower * (height_m ** 2)
    target_weight_upper = normal_bmi_upper * (height_m ** 2)

    # Calculate BMI and determine category
    bmi = user["Weight"] / (height_m ** 2)
    if bmi < normal_bmi_lower:
        bmi_category = "Underweight"
    elif normal_bmi_lower <= bmi <= normal_bmi_upper:
        bmi_category = "Normal weight"
    elif normal_bmi_upper < bmi <= 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # Calculate weight adjustments
    weight_to_lose = max(0, user["Weight"] - target_weight_upper)
    weight_to_gain = max(0, target_weight_lower - user["Weight"])

    # Initialize weight prediction logic
    weight_predictions = [user["Weight"]]
    if weight_to_lose > 0:
        for i in range(1, 13):
            predicted_weight = user["Weight"] - i * (weight_to_lose / 12)
            weight_predictions.append(predicted_weight)
    elif weight_to_gain > 0:
        for i in range(1, 13):
            predicted_weight = user["Weight"] + i * (weight_to_gain / 12)
            weight_predictions.append(predicted_weight)
    else:
        for i in range(1, 13):
            weight_predictions.append(user["Weight"])

    # Create the graph
    weeks = list(range(0, 13))
    weight_graph = go.Figure()
    weight_graph.add_trace(go.Scatter(x=[0], y=user_weight, mode="lines+markers", name="Weight History"))
    weight_graph.add_trace(go.Scatter(x=weeks, y=weight_predictions, mode="lines", name="Predicted Weight"))
    weight_graph.add_trace(go.Scatter(x=weeks, y=[target_weight_lower] * len(weeks), mode="lines", name="Target Weight (Lower BMI)", line=dict(dash="dash", color="green")))
    weight_graph.add_trace(go.Scatter(x=weeks, y=[target_weight_upper] * len(weeks), mode="lines", name="Target Weight (Upper BMI)", line=dict(dash="dash", color="green")))

    # Add weight annotations
    weight_graph.add_annotation(
        x=0, y=user["Weight"], text=f"Current Weight: {user['Weight']} kg", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(size=12)
    )
    if weight_to_lose > 0:
        weight_graph.add_annotation(
            x=weeks[-1], y=user["Weight"] - weight_to_lose, text=f"Need to lose {round(weight_to_lose, 2)} kg", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(size=12, color="red")
        )
    if weight_to_gain > 0:
        weight_graph.add_annotation(
            x=weeks[-1], y=user["Weight"] + weight_to_gain, text=f"Need to gain {round(weight_to_gain, 2)} kg", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(size=12, color="blue")
        )

    weight_graph.update_layout(title="Weight Tracking and Prediction", xaxis_title="Weeks", yaxis_title="Weight (kg)", showlegend=True)
    weight_graph_json = json.dumps(weight_graph, cls=plotly.utils.PlotlyJSONEncoder)

    # Recommendations based on BMI category
    recommendations = {}
    if bmi_category == "Underweight":
        recommendations = {
            "Goal": "Consume more calories than burned, focusing on calorie surplus.",
            "Daily Caloric Intake": "2500–3000 kcal for men, 2200–2500 kcal for women.",
            "Weekly Exercise Plan": [
                "Day 1: Strength Training (Upper Body) - 60 mins",
                "Day 2: Strength Training (Lower Body) - 60 mins",
                "Day 3: Active Recovery (Yoga or Light Stretching) - 30 mins",
                "Day 4: Strength Training (Full Body) - 60 mins",
                "Day 5: Strength Training (Core and Arms) - 60 mins",
                "Day 6: Cardio (Low Intensity, e.g., Walking/Cycling) - 30 mins",
                "Day 7: Rest Day"
            ],
            "Recommended Foods": [
                "High-quality proteins: Eggs, chicken, salmon, whey protein shakes",
                "Carbs: Sweet potatoes, pasta, bread, rice",
                "Fats: Peanut butter, olive oil, cheese",
                "Dairy: Whole milk, Greek yogurt",
                "Snacks: Granola bars, protein bars, smoothies"
            ]    
        }
        exercises = ["Strength Training (Upper Body)", "Strength Training (Lower Body)", "Active Recovery (Yoga)", "Strength Training(Full Body))","Strength Training (Core and Arms)","Cardio (Low Intensity)"]
    elif bmi_category == "Normal weight":
        recommendations = {
            "Goal": "Maintain calorie intake at a level that balances energy expenditure.",
            "Daily Caloric Intake": "~2000 kcal for women, ~2500 kcal for men.",
             "Weekly Exercise Plan": [
                "Day 1: Cardio (Jogging or Swimming) - 45 mins",
                "Day 2: Strength Training (Full Body) - 60 mins",
                "Day 3: Active Recovery (Yoga) - 30 mins",
                "Day 4: Strength Training (Upper and Core) - 60 mins",
                "Day 5: Cardio (Cycling or Rowing) - 45 mins",
                "Day 6: Strength Training (Lower Body) - 60 mins",
                "Day 7: Rest Day"
            ],
            "Recommended Foods": [
                "Balanced meals: 40% carbs, 30% proteins, 30% fats",
                "Proteins: Chicken, fish, eggs, beans",
                "Carbs: Whole grains, sweet potatoes, fruits",
                "Fats: Nuts, seeds, avocado"
            ]
        }
        exercises = ["Cardio (Jogging or Swimming)", "Strength Training (Full Body)", "Active Recovery (Yoga)", "Strength Training (Upper and Core)", "Cardio (Cycling or Rowing)", "Strength Training (Lower Body)"]
    elif bmi_category == "Overweight" or bmi_category == "Obese":
        recommendations = {
            "Goal": "Burn more calories than consumed, focusing on calorie deficit.",
            "Daily Caloric Intake": "1200–1500 kcal for women, 1500–1800 kcal for men.",
           "Weekly Exercise Plan": [
                "Day 1: Cardio (e.g., Running/Brisk Walking) - 45 mins",
                "Day 2: Strength Training (Full Body with a focus on compound exercises) - 60 mins",
                "Day 3: High-Intensity Interval Training (HIIT) - 30 mins",
                "Day 4: Active Recovery (Yoga or Stretching) - 30 mins",
                "Day 5: Cardio (Cycling/Swimming) - 45 mins",
                "Day 6: Strength Training (Focus on Core and Lower Body) - 60 mins",
                "Day 7: Rest Day"
            ],
            "Recommended Foods": [
                "Lean proteins: Chicken breast, tofu, fish",
                "Whole grains: Brown rice, quinoa, oats",
                "Vegetables: Spinach, broccoli, kale, bell peppers",
                "Fruits: Apples, berries, oranges",
                "Healthy fats: Avocado, nuts, seeds"
            ]
        }
        exercises = ["Cardio (Running/Brisk Walking)", "Strength Training (Full Body)", "HIIT", "Active Recovery (Yoga)", "Cardio (Cycling/Swimming)", "Strength Training (Core and Lower Body)"]

    user_logs = []
    with open('exercise_logs.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:  # Check if the log belongs to the current user
                user_logs.append(row)

    last_recorded_weight = user["Weight"]  # Default to current weight if no logs exist
    if user_logs:
        last_recorded_weight = float(user_logs[-1][5])  # Assuming column 5 is 'WeightAfter'

    weight_change_rate = 0.5  # kg per week (assume 0.5 kg for both gain and loss for now)

    # Calculate the total number of weeks required
    if last_recorded_weight > target_weight_upper:
        total_weeks = int((last_recorded_weight - target_weight_upper) / weight_change_rate)
        target_reached = "lose weight"
    elif last_recorded_weight < target_weight_lower:
        total_weeks = int((target_weight_lower - last_recorded_weight) / weight_change_rate)
        target_reached = "gain weight"
    else:
        total_weeks = 0
        target_reached = "maintain weight"

    # Generate weight predictions dynamically
    weeks = list(range(total_weeks + 1))
    predicted_weights_last_recorded = [last_recorded_weight]

    for i in range(1, total_weeks + 1):
        if target_reached == "lose weight":
            predicted_weights_last_recorded.append(last_recorded_weight - i * weight_change_rate)
        elif target_reached == "gain weight":
            predicted_weights_last_recorded.append(last_recorded_weight + i * weight_change_rate)
        else:
            predicted_weights_last_recorded.append(last_recorded_weight)

    # Create the second graph
    second_graph = go.Figure()
    second_graph.add_trace(go.Scatter(
        x=weeks,
        y=predicted_weights_last_recorded,
        mode="lines+markers",
        name=f"Predicted Weight ({target_reached})"
    ))
    second_graph.add_trace(go.Scatter(
        x=weeks,
        y=[target_weight_lower] * len(weeks),
        mode="lines",
        name="Target Weight (Lower BMI)",
        line=dict(dash="dash", color="green")
    ))
    second_graph.add_trace(go.Scatter(
        x=weeks,
        y=[target_weight_upper] * len(weeks),
        mode="lines",
        name="Target Weight (Upper BMI)",
        line=dict(dash="dash", color="green")
    ))

    second_graph.update_layout(
        title=f"Predicted Weight ({target_reached}) Over Time",
        xaxis_title="Weeks",
        yaxis_title="Weight (kg)",
        showlegend=True
    )

    second_graph_json = json.dumps(second_graph, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "user_dashboard.html",
        user=user,
        weight_graph_json=weight_graph_json,
        second_graph_json=second_graph_json,
        bmi_category=bmi_category,
        recommendations=recommendations,
        exercises=exercises
    )


@app.route("/user/<name>/exercise_logs", methods=["GET", "POST"])
def exercise_logs(name):
    # Initialize user logs list
    user_logs = []
    
    # Try to read from the exercise logs file
    try:
        with open(exercise_logs_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == name:  # Check if the log belongs to the current user
                    user_logs.append(row)
    except FileNotFoundError:
        # Handle case where the file does not exist
        print("Exercise logs file not found. A new file will be created upon saving.")
    except PermissionError as e:
        return f"Permission denied: {e}", 403

    if request.method == "POST":
        # Validate and save exercise log for the user
        date = request.form["date"]
        exercise_type = request.form["exercise_type"]

        # Input validation
        try:
            duration = int(request.form["duration"])
            weight_before = float(request.form["weight_before"])
            weight_after = float(request.form["weight_after"])
        except ValueError:
            return "Invalid input. Please ensure all fields are correctly filled.", 400

        # Attempt to save log to CSV file
        try:
            with open(exercise_logs_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, date, exercise_type, duration, weight_before, weight_after])
        except PermissionError as e:
            return f"Permission denied when writing to the file: {e}", 403

        return redirect(url_for("exercise_logs", name=name))

    # Render the exercise log page
    return render_template(
        "exercise_logs.html",
        user=name,
        user_logs=user_logs,
        exercises=[
            "Active Recovery (Yoga)",
            "Cardio (Cycling/Rowing)",
            "Cardio (Cycling/Swimming)",
            "Cardio (Jogging or Swimming)",
            "Cardio (Low Intensity)",
            "Cardio (Running/Brisk Walking)",
            "HIIT",
            "Strength Training (Core and Arms)",
            "Strength Training (Core and Lower Body)",
            "Strength Training (Full Body)",
            "Strength Training (Lower Body)",
            "Strength Training (Upper and Core)",
            "Strength Training (Upper Body)",
            "Rest"
        ]
    )


@app.route("/user/<name>/prediction_graph")
def prediction_graph(name):
    user_logs = []
    with open(exercise_logs_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:
                user_logs.append(row)

    if len(user_logs) < 2:
        return "Not enough data to generate predictions.", 400

    user = users_data[users_data["Name"] == name].iloc[0]
    initial_weight = user["Weight"]

    # Calculate predictions
    days, predicted_weights = calculate_predictions(user_logs, initial_weight)

    # Create the graph
    prediction_graph = go.Figure()
    prediction_graph.add_trace(go.Scatter(
        x=days,
        y=predicted_weights,
        mode="lines+markers",
        name="Predicted Weight"
    ))
    prediction_graph.update_layout(
        title="Weight Prediction for Next Week",
        xaxis_title="Days from Start",
        yaxis_title="Weight (kg)"
    )

    # Pass graph JSON to template
    graph_json = json.dumps(prediction_graph, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("prediction_graph.html", user=name, graph_json=graph_json)

@app.route("/user/<name>/exercise_intensity", methods=["GET"])
def exercise_intensity(name):
    # Read the user logs from the CSV file
    user_logs = []
    with open(exercise_logs_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:  # Filter logs by the current user
                user_logs.append(row)

    # Generate the exercise intensity graphs
    graphs = generate_exercise_intensity_graphs(user_logs)

    # Render the template with the generated graphs
    return render_template(
        "exercise_intensity.html",
        user=name,
        graphs=graphs
    )

@app.route("/user/<name>/prediction_graph_updated")
def prediction_graph_updated(name):
    user_logs = []
    with open(exercise_logs_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:
                user_logs.append(row)

    if len(user_logs) < 2:
        return "Not enough data to generate predictions.", 400

    user = users_data[users_data["Name"] == name].iloc[0]
    height_m = user["Height"] / 100
    normal_bmi_lower = 18.5
    normal_bmi_upper = 24.9
    target_weight = (normal_bmi_lower + normal_bmi_upper) / 2 * (height_m ** 2)

    # Get the latest weight from exercise logs
    last_weight = float(user_logs[-1][5])  # Weight_After column

    # Calculate predictions
    days, predicted_weights, weeks_to_normal_bmi = calculate_predictions_based_on_logs(user_logs, last_weight, target_weight)

    # Create the graph
    prediction_graph = go.Figure()
    prediction_graph.add_trace(go.Scatter(
        x=days,
        y=predicted_weights,
        mode="lines+markers",
        name="Predicted Weight"
    ))
    prediction_graph.add_trace(go.Scatter(
        x=[days[0], days[-1]],
        y=[target_weight, target_weight],
        mode="lines",
        name="Target Weight (Normal BMI)",
        line=dict(dash="dash", color="green")
    ))
    prediction_graph.update_layout(
        title=f"Prediction from Last Recorded Weight ({last_weight} kg)",
        xaxis_title="Days from Start",
        yaxis_title="Weight (kg)"
    )

    # Add annotation for weeks to normal BMI
    if weeks_to_normal_bmi is not None:
        prediction_graph.add_annotation(
            x=days[-1],
            y=target_weight,
            text=f"Estimated time to reach normal BMI: {weeks_to_normal_bmi} weeks",
            showarrow=False,
            font=dict(color="blue", size=12)
        )

    # Pass graph JSON to template
    graph_json = json.dumps(prediction_graph, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("prediction_graph_updated.html", user=name, graph_json=graph_json)

@app.route("/user/<name>/weight_prediction_graph", methods=["GET"])
def weight_prediction_graph(name):
    # Fetch user details
    user = users_data[users_data["Name"] == name].iloc[0]
    height_m = user["Height"] / 100

    # Calculate target weight range (BMI 18.5 - 24.9)
    normal_bmi_lower = 18.5
    normal_bmi_upper = 24.9
    target_weight_lower = normal_bmi_lower * (height_m ** 2)
    target_weight_upper = normal_bmi_upper * (height_m ** 2)

    # Fetch user's exercise logs
    user_logs = []
    with open(exercise_logs_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:
                user_logs.append(row)

    if len(user_logs) < 2:
        return "Not enough data to generate predictions.", 400

    # Extract the last recorded weight from logs
    last_recorded_weight = float(user_logs[-1][5])  # Assuming 'WeightAfter' is column 5

    # Determine target weight based on BMI category
    if last_recorded_weight > target_weight_upper:
        target_weight = target_weight_upper  # Weight loss required
    elif last_recorded_weight < target_weight_lower:
        target_weight = target_weight_lower  # Weight gain required
    else:
        target_weight = last_recorded_weight  # Already in normal BMI range

    # Get predictions
    days, predicted_weights, _ = calculate_weight_reduction_weeks(user_logs, last_recorded_weight, target_weight)

    # Truncate predictions to stop at the BMI target
    truncated_days = []
    truncated_weights = []
    for day, weight in zip(days, predicted_weights):
        if (last_recorded_weight > target_weight and weight <= target_weight) or \
           (last_recorded_weight < target_weight and weight >= target_weight):
            truncated_days.append(day)
            truncated_weights.append(target_weight)  # Stop at the target
            break
        truncated_days.append(day)
        truncated_weights.append(weight)

    # Create the graph
    prediction_graph = go.Figure()
    prediction_graph.add_trace(go.Scatter(
        x=truncated_days,
        y=truncated_weights,
        mode="lines+markers",
        name="Predicted Weight"
    ))

    # Add horizontal line for the target BMI weight
    prediction_graph.add_trace(go.Scatter(
        x=[truncated_days[0], truncated_days[-1]],
        y=[target_weight, target_weight],
        mode="lines",
        name="Target Weight (BMI)",
        line=dict(dash="dash", color="green")
    ))

    prediction_graph.update_layout(
        title=f"Weight Prediction for {name}",
        xaxis_title="Days",
        yaxis_title="Weight (kg)",
        showlegend=True
    )

    # Annotate the point where target BMI is reached
    prediction_graph.add_annotation(
        x=truncated_days[-1],
        y=target_weight,
        text=f"Target Weight Reached: {target_weight:.2f} kg",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="blue", size=12)
    )

    graph_json = json.dumps(prediction_graph, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("weight_prediction_graph.html", user=name, graph_json=graph_json)








@app.route("/new_user", methods=["GET", "POST"])
def new_user():
    if request.method == "POST":
        # Add new user
        name = request.form["name"]
        age = int(request.form["age"])
        gender = request.form["gender"]
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        activity_level = request.form["activity_level"]

        bmi = calculate_bmi(weight, height)
        new_user_data = pd.DataFrame([{
            "Name": name, "Age": age, "Gender": gender,
            "Weight": weight, "Height": height, "BMI": bmi, "ActivityLevel": activity_level
        }])
        global users_data
        users_data = pd.concat([users_data, new_user_data], ignore_index=True)
        users_data.to_csv(users_file, index=False)

        return redirect(url_for("user_dashboard", name=name))

    return render_template("new_user.html")

if __name__ == "__main__":
    app.run(debug=True)
