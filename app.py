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
        "Naive Bayes": "mushroom_naive_bayes.joblib",
        "K Nearest Neighbor": "mushroom_knn(22).joblib"
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

feature_labels = {
    "cap-shape": "Hình dạng mũ nấm",
    "cap-surface": "Bề mặt mũ nấm",
    "cap-color": "Màu sắc mũ nấm",
    "bruises": "Bầm dập",
    "odor": "Mùi",
    "gill-attachment": "Kiểu gắn kết phiến nấm",
    "gill-spacing": "Khoảng cách giữa các phiến nấm",
    "gill-size": "Kích thước phiến nấm",
    "gill-color": "Màu sắc phiến nấm",
    "stalk-shape": "Hình dạng cuống nấm",
    "stalk-root": "Hình dạng gốc cuống nấm",
    "stalk-surface-above-ring": "Bề mặt cuống phía trên vòng",
    "stalk-surface-below-ring": "Bề mặt cuống phía dưới vòng",
    "stalk-color-above-ring": "Màu sắc cuống phía trên vòng",
    "stalk-color-below-ring": "Màu sắc cuống phía dưới vòng",
    "veil-type": "Kiểu màng nấm",
    "veil-color": "Màu sắc màng nấm",
    "ring-number": "Số lượng vòng trên cuống",
    "ring-type": "Kiểu vòng trên cuống",
    "spore-print-color": "Màu sắc bào tử",
    "population": "Mật độ phân bố",
    "habitat": "Môi trường sống"
}

labels = {
    "cap-shape": {
        "b": "Hình chuông",
        "c": "Hình nón",
        "x": "Lồi",
        "f": "Phẳng",
        "k": "Có núm",
        "s": "Lõm"
    },
    "cap-surface": {
        "f": "Dạng sợi",
        "g": "Có rãnh",
        "y": "Có vảy",
        "s": "Nhẵn"
    },
    "cap-color": {
        "n": "Nâu",
        "b": "Vàng nhạt",
        "c": "Nâu quế",
        "g": "Xám",
        "r": "Xanh lục",
        "p": "Hồng",
        "u": "Tím",
        "e": "Đỏ",
        "w": "Trắng",
        "y": "Vàng"
    },
    "bruises": {
        "t": "Có vết bầm",
        "f": "Không có vết bầm"
    },
    "odor": {
        "a": "Mùi hạnh nhân",
        "l": "Mùi hồi",
        "c": "Mùi hắc ín",
        "y": "Mùi tanh",
        "f": "Mùi khó chịu",
        "m": "Mùi mốc",
        "n": "Không có mùi",
        "p": "Mùi hăng",
        "s": "Mùi cay"
    },
    "gill-attachment": {
        "a": "Gắn liền",
        "d": "Chạy dọc xuống cuống",
        "f": "Tự do",
        "n": "Có khía"
    },
    "gill-spacing": {
        "c": "Sát nhau",
        "w": "Dày đặc",
        "d": "Thưa"
    },
    "gill-size": {
        "b": "Rộng",
        "n": "Hẹp"
    },
    "gill-color": {
        "k": "Đen",
        "n": "Nâu",
        "b": "Vàng nhạt",
        "h": "Nâu sô-cô-la",
        "g": "Xám",
        "r": "Xanh lục",
        "o": "Cam",
        "p": "Hồng",
        "u": "Tím",
        "e": "Đỏ",
        "w": "Trắng",
        "y": "Vàng"
    },
    "stalk-shape": {
        "e": "Phình to về phía gốc",
        "t": "Thuôn nhỏ về phía gốc"
    },
    "stalk-root": {
        "b": "Dạng củ",
        "c": "Dạng chùy",
        "u": "Dạng cốc",
        "e": "Đều",
        "z": "Dạng thân rễ",
        "r": "Có rễ",
        "missing": "Không xác định"
    },
    "stalk-surface-above-ring": {
        "f": "Dạng sợi",
        "y": "Có vảy",
        "k": "Mượt như lụa",
        "s": "Nhẵn"
    },
    "stalk-surface-below-ring": {
        "f": "Dạng sợi",
        "y": "Có vảy",
        "k": "Mượt như lụa",
        "s": "Nhẵn"
    },
    "stalk-color-above-ring": {
        "n": "Nâu",
        "b": "Vàng nhạt",
        "c": "Nâu quế",
        "g": "Xám",
        "o": "Cam",
        "p": "Hồng",
        "e": "Đỏ",
        "w": "Trắng",
        "y": "Vàng"
    },
    "stalk-color-below-ring": {
        "n": "Nâu",
        "b": "Vàng nhạt",
        "c": "Nâu quế",
        "g": "Xám",
        "o": "Cam",
        "p": "Hồng",
        "e": "Đỏ",
        "w": "Trắng",
        "y": "Vàng"
    },
    "veil-type": {
        "p": "Màng riêng",
        "u": "Màng chung"
    },
    "veil-color": {
        "n": "Nâu",
        "o": "Cam",
        "w": "Trắng",
        "y": "Vàng"
    },
    "ring-number": {
        "n": "Không có vòng",
        "o": "Một vòng",
        "t": "Hai vòng"
    },
    "ring-type": {
        "c": "Dạng mạng nhện",
        "e": "Dễ biến mất",
        "f": "Xòe rộng",
        "l": "Lớn",
        "n": "Không có vòng",
        "p": "Dạng rủ",
        "s": "Dạng bao bọc",
        "z": "Dạng vòng đai"
    },
    "spore-print-color": {
        "k": "Đen",
        "n": "Nâu",
        "b": "Vàng nhạt",
        "h": "Nâu sô-cô-la",
        "r": "Xanh lục",
        "o": "Cam",
        "u": "Tím",
        "w": "Trắng",
        "y": "Vàng"
    },
    "population": {
        "a": "Rất nhiều",
        "c": "Mọc thành cụm",
        "n": "Số lượng lớn",
        "s": "Phân tán",
        "v": "Một số",
        "y": "Mọc đơn lẻ"
    },
    "habitat": {
        "g": "Đồng cỏ",
        "l": "Lá cây",
        "m": "Đồng cỏ tự nhiên",
        "p": "Đường mòn",
        "u": "Khu vực đô thị",
        "w": "Khu vực chất thải",
        "d": "Rừng"
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
                feature_labels[feature],
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

        # Lấy danh sách đặc trưng đã sử dụng khi huấn luyện
        if hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)

        elif hasattr(model, "named_steps"):
            preprocessor = model.named_steps["preprocessor"]

            if hasattr(preprocessor, "feature_names_in_"):
                feature_columns = list(preprocessor.feature_names_in_)
            else:
                raise ValueError(
                    "Không tìm thấy danh sách đặc trưng đã huấn luyện."
                )
        else:
            raise ValueError(
                "Mô hình không lưu thông tin tên đặc trưng."
            )

        # Lọc các đặc trưng không sử dụng, ví dụ veil-type
        sample = sample.reindex(columns=feature_columns)

        # Dự đoán
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