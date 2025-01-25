---

# 🌟 **E`STATE**: Elevate Room Layout Predictions to the Next Dimension 🌌

### Predict | Visualize | Innovate 🚀

Mestri is a cutting-edge, AI-powered platform for transforming **RGB panorama images** into **stunning 3D room layouts**. Designed for architects, interior designers, AR/VR enthusiasts, and AI researchers, Mestri uses state-of-the-art **Geometry-Aware Transformer Networks (LGT-Net)** to generate immersive visualizations that redefine what's possible in 3D room modeling.

---

## ✨ **Key Features**
- 🔮 **3D Room Layout Prediction**: Unlock the power of **LGT-Net** to generate ultra-precise 3D room layouts.
- 🎯 **Customizable Pre-Processing**: Choose your preferred pipeline settings for unparalleled flexibility.
- 🎨 **Multi-Format Outputs**: Export results in `.gltf`, `.obj`, `.glb`, or stunning 2D floorplans.
- 📊 **Depth & Gradient Visualizations**: Explore insightful visual representations of room geometries.
- 🌐 **API-Ready**: Integrates seamlessly with frontend applications via RESTful endpoints.
- ⚙️ **Cross-Platform Interfaces**: Enjoy flexibility with both **Streamlit** and **Gradio** interfaces.

---

## 🛠️ **Installation**

### 📋 **Prerequisites**
Ensure you have the following:
- **Python**: Version 3.8 or higher
- **Pip**: For managing dependencies
- **Git**: To clone the repository

### 🔧 **Clone the Repository**
```bash
git clone https://github.com/fardeenKhadri/mestri.git
cd mestri
```

### 📦 **Install Dependencies**
```bash
pip install -r requirements.txt
```

### ⬇️ **Download Pretrained Models**
Automatically fetch pretrained models:
```bash
python app.py
```

---

## 🚀 **Getting Started**

### 🌟 **1. Streamlit Interface**
Run the Streamlit app to explore Mestri's features:
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```
- **Access**: [http://localhost:8501](http://localhost:8501)
- **What to Expect**:
  - Upload an **RGB Panorama**.
  - View real-time **predictions** and **3D visualizations**.
  - Download **results** for further processing.

---

### 🎨 **2. Gradio Interface**
For a sleek, browser-based experience:
```bash
python app.py
```
- **What to Expect**:
  - A **drag-and-drop** interface for quick predictions.
  - Support for depth-normal-gradient, floorplans, and `.gltf` exports.

---

### 🌐 **3. REST API for Integration**
Integrate Mestri into your own workflows via REST endpoints!

#### 🔗 **Endpoint**
```http
POST /api/vrshow
```

#### 🛠️ **Example Request**
```bash
curl -X POST -F "image=@path_to_image.jpg" http://localhost:5000/api/vrshow
```

#### 📦 **Response**
- **Success**:
  ```json
  {
    "success": true,
    "result_image": "path_to_predicted_image.png"
  }
  ```
- **Error**:
  ```json
  {
    "error": "Error message"
  }
  ```

---

## ⚙️ **Configuration**

### 🔧 `config/defaults.py`
Easily configure:
- **Model Architecture**: Select between `Transformer` and `LSTM`.
- **Output Formats**: Choose `.gltf`, `.obj`, `.glb`.
- **Pre-Processing**: Enable or disable as needed.

---

## 📂 **Project Structure**

```
mestri/
├── app.py                 # Gradio-based interface
├── streamlit_app.py       # Streamlit-based interface
├── src/
│   ├── config/
│   │   └── defaults.py    # Model and app configuration
│   ├── models/
│   │   ├── build.py       # Model builder logic
│   │   ├── lgt_net.py     # LGT-Net architecture
│   ├── utils/             # Helper functions
│   ├── output/            # Output folder for predictions
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🌟 **How It Works**

1. **📥 Upload Image**: Provide an RGB panorama through the interface or API.
2. **🤖 Model Inference**: `LGT-Net` processes the image to predict the room layout.
3. **📊 Visualize Results**: Explore the 2D visualizations and 3D reconstructions.
4. **📤 Export Outputs**: Save results in `.gltf`, `.obj`, `.glb`, or JSON formats.

---

## 🚀 **Example Workflow**

1. **Input**: Upload an image of your room or space.
2. **Processing**: Mestri's **Transformer**-based decoder predicts the layout.
3. **Outputs**:
   - 2D visualizations: **Depth-normal-gradient** and **Floorplans**.
   - 3D Mesh: Downloadable `.gltf` or `.obj` files.

---

## 🔥 **Future Enhancements**

- **💡 Real-Time Visualizations**: Enable real-time 3D rendering.
- **🌎 Global Dataset Compatibility**: Extend support for diverse datasets.
- **📱 Mobile App**: Launch a companion mobile application.
- **🎨 Advanced Post-Processing**: Add photorealistic rendering for AR/VR.

---

## 🤝 **Contributing**

We love contributions! 💖 Here’s how you can help:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request on GitHub.

---

## 📄 **License**

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 **Acknowledgments**

- **`LGT-Net`**: For inspiring this project's incredible results.
- **Community**: Thanks to all contributors and users who support this initiative.

---

## 🌌 **Explore the Future of 3D Room Layout Predictions with Mestri!**
Let Mestri empower your workflows, from design to immersive visualization. 🚀

---

