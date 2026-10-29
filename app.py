import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

@st.cache_resource
def load_models():
    models = {}

    model_files = {
        "Decision Tree": "mushroom_decision_tree.joblib",
        "Random Forest": "mushroom_random_forest.joblib",
        "K Nearest Neighbor": "mushroom_knn.joblib"
    }

    for name, filename in model_files.items():
        path = Path(filename)
        if path.exists():
            models[name] = joblib.load(path)

    return models


st.title("Mushroom Classification")
st.write(
    "Nhập các đặc tính của mẫu nấm và chọn mô hình "
    "đã huấn luyện để xem kết quả dự đoán."
)

models = load_models()

if not models:
    st.error(
        "Không tìm thấy file mô hình. "
        "Hãy đặt các file .joblib cùng thư mục với app.py "
        "hoặc sửa lại đường dẫn trong code."
    )
    st.stop()

options = {
    "cap-shape": ["b", "c", "x", "f", "k", "s"],
    "cap-surface": ["f", "g", "y", "s"],
    "cap-color": ["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"],
    "bruises": ["t", "f"],
    "odor": ["a", "l", "c", "y", "f", "m", "n", "p", "s"],
    "gill-attachment": ["a", "d", "f", "n"],
    "gill-spacing": ["c", "w", "d"],
    "gill-size": ["b", "n"],
    "gill-color": ["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"],
    "stalk-shape": ["e", "t"],
    "stalk-root": ["b", "c", "u", "e", "z", "r", "missing"],
    "stalk-surface-above-ring": ["f", "y", "k", "s"],
    "stalk-surface-below-ring": ["f", "y", "k", "s"],
    "stalk-color-above-ring": ["n", "b", "c", "g", "o", "p", "e", "w", "y"],
    "stalk-color-below-ring": ["n", "b", "c", "g", "o", "p", "e", "w", "y"],
    "veil-type": ["p", "u"],
    "veil-color": ["n", "o", "w", "y"],
    "ring-number": ["n", "o", "t"],
    "ring-type": ["c", "e", "f", "l", "n", "p", "s", "z"],
    "spore-print-color": ["k", "n", "b", "h", "r", "o", "u", "w", "y"],
    "population": ["a", "c", "n", "s", "v", "y"],
    "habitat": ["g", "l", "m", "p", "u", "w", "d"]
}


labels = {
    "cap-shape": {
        "b": "Bell", "c": "Conical", "x": "Convex",
        "f": "Flat", "k": "Knobbed", "s": "Sunken"
    },
    "cap-surface": {
        "f": "Fibrous", "g": "Grooves",
        "y": "Scaly", "s": "Smooth"
    },
    "cap-color": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon",
        "g": "Gray", "r": "Green", "p": "Pink",
        "u": "Purple", "e": "Red", "w": "White",
        "y": "Yellow"
    },
    "bruises": {
        "t": "Bruises", "f": "No bruises"
    },
    "odor": {
        "a": "Almond", "l": "Anise", "c": "Creosote",
        "y": "Fishy", "f": "Foul", "m": "Musty",
        "n": "None", "p": "Pungent", "s": "Spicy"
    },
    "gill-attachment": {
        "a": "Attached", "d": "Descending",
        "f": "Free", "n": "Notched"
    },
    "gill-spacing": {
        "c": "Close", "w": "Crowded", "d": "Distant"
    },
    "gill-size": {
        "b": "Broad", "n": "Narrow"
    },
    "gill-color": {
        "k": "Black", "n": "Brown", "b": "Buff",
        "h": "Chocolate", "g": "Gray", "r": "Green",
        "o": "Orange", "p": "Pink", "u": "Purple",
        "e": "Red", "w": "White", "y": "Yellow"
    },
    "stalk-shape": {
        "e": "Enlarging", "t": "Tapering"
    },
    "stalk-root": {
        "b": "Bulbous", "c": "Club", "u": "Cup",
        "e": "Equal", "z": "Rhizomorphs",
        "r": "Rooted", "missing": "Missing / Unknown"
    },
    "stalk-surface-above-ring": {
        "f": "Fibrous", "y": "Scaly",
        "k": "Silky", "s": "Smooth"
    },
    "stalk-surface-below-ring": {
        "f": "Fibrous", "y": "Scaly",
        "k": "Silky", "s": "Smooth"
    },
    "stalk-color-above-ring": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon",
        "g": "Gray", "o": "Orange", "p": "Pink",
        "e": "Red", "w": "White", "y": "Yellow"
    },
    "stalk-color-below-ring": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon",
        "g": "Gray", "o": "Orange", "p": "Pink",
        "e": "Red", "w": "White", "y": "Yellow"
    },
    "veil-type": {
        "p": "Partial", "u": "Universal"
    },
    "veil-color": {
        "n": "Brown", "o": "Orange",
        "w": "White", "y": "Yellow"
    },
    "ring-number": {
        "n": "None", "o": "One", "t": "Two"
    },
    "ring-type": {
        "c": "Cobwebby", "e": "Evanescent",
        "f": "Flaring", "l": "Large", "n": "None",
        "p": "Pendant", "s": "Sheathing", "z": "Zone"
    },
    "spore-print-color": {
        "k": "Black", "n": "Brown", "b": "Buff",
        "h": "Chocolate", "r": "Green", "o": "Orange",
        "u": "Purple", "w": "White", "y": "Yellow"
    },
    "population": {
        "a": "Abundant", "c": "Clustered",
        "n": "Numerous", "s": "Scattered",
        "v": "Several", "y": "Solitary"
    },
    "habitat": {
        "g": "Grasses", "l": "Leaves", "m": "Meadows",
        "p": "Paths", "u": "Urban", "w": "Waste",
        "d": "Woods"
    }
}

model_name = st.selectbox(
    "Chọn mô hình",
    list(models.keys())
)

st.subheader("Nhập đặc tính của mẫu nấm")
st.caption(
    "Các ký hiệu là mã phân loại gốc của dataset UCI Mushroom."
)

with st.form("mushroom_form"):
    input_values = {}

    feature_columns = list(options.keys())
    columns = st.columns(6)

    for i, feature in enumerate(feature_columns):
        with columns[i % 6]:
            default_index = 0

            if feature == "odor":
                default_index = options[feature].index("n")

            if feature == "stalk-root":
                default_index = options[feature].index("missing")

            input_values[feature] = st.selectbox(
                feature,
                options[feature],
                format_func=lambda code, f=feature: labels[f][code],
                index=default_index,
                key=feature
            )

    submitted = st.form_submit_button(
        "Dự đoán",
        type="primary",
        use_container_width=True
    )

if submitted:
    sample = pd.DataFrame([input_values])

    try:
        model = models[model_name]

        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        classes = model.classes_

        st.divider()
        st.subheader("Kết quả dự đoán")

        if prediction == 0:
            st.success("Mô hình dự đoán: Edible")
        else:
            st.error("Mô hình dự đoán: Poisonous")

        probability_data = pd.DataFrame({
            "Nhãn": [
                "Edible (0)" if int(c) == 0 else "Poisonous (1)"
                for c in classes
            ],
            "Xác suất dự đoán": probabilities
        })

        probability_data["Xác suất (%)"] = (
            probability_data["Xác suất dự đoán"] * 100
        )

        st.write("Xác suất theo mô hình:")
        st.dataframe(
            probability_data[["Nhãn", "Xác suất (%)"]],
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Không thể dự đoán: {e}")

