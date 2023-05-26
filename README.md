# Prophet Forecasting App <a href="https://prophet-forecasting-app.onrender.com" target="_parent"><img src="images/play-button-arrowhead.png" style="width:18px" alt="Open On Render"/></a>

A Streamlit WebApp.

## **How to use this App?**
### **Clone this repository**
```bash
  git clone https://github.com/utkarshbelkhede/Prophet_Forecasting_App.git
```
### **Method 1 - Using Docker**
**Prerequisite** - Docker

1. Build Docker Images
```bash
  docker build -t prophetforecastingapp:latest .
```
2. Run App in Docker Container
```bash
  docker run prophetforecastingapp:latest
```
### **Method 2 - Using Virtual Environment**
**Prerequisite** - Conda, Python
1. Create a Conda Virtual Environment
```bash
  conda create -n prophetforecastingapp
```
2. Activate Virtual Environment
```bash
  conda activate prophetforecastingapp
```
3. Install Requirements.txt
```bash
  pip install -r requirements.txt
```
4. Run Streamlit App
```bash
  streamlit run app.py
```
