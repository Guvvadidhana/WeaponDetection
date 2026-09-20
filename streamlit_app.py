import streamlit as st
import cv2
import numpy as np

st.title("🔫 Weapon Detection System")

@st.cache_resource
def load_model():
    model = cv2.dnn.readNet(
        "model/weapon_training_2000.weights",
        "model/weapon_testing.cfg"
    )
    return model

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image"
    )

    if st.button("Detect Weapon"):

        model = load_model()

        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(
            img,
            0.00392,
            (416, 416),
            (0, 0, 0),
            True,
            crop=False
        )

        model.setInput(blob)

        layer_names = model.getLayerNames()
        output_layers = [
            layer_names[i - 1]
            for i in model.getUnconnectedOutLayers().flatten()
        ]

        outputs = model.forward(output_layers)

        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:

                scores = detection[5:]
                confidence = float(np.max(scores))

                if confidence > 0.5:

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)

        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            0.5,
            0.4
        )

        if len(indexes) > 0:

            for i in np.array(indexes).flatten():

                x, y, w, h = boxes[i]

                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    "Weapon",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            st.error("⚠️ Weapon Detected!")

        else:
            st.success("✅ No Weapon Detected")

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Detection Result"
        )
