# PA_by_Transformers

## 🧠 GPT Transformer from Scratch with Fine-tuning & Web App Deployment

This project demonstrates the **end-to-end creation of a GPT-like architecture from scratch**, integrating:
- Custom-built Transformer model in PyTorch
- Loading **pretrained GPT-2 weights**
- Fine-tuning on an **instruction-following dataset** in Alpaca format
- Deploying via a **Flask web app** for interactive testing

---

## 📌 Features
- **Transformer-based GPT model** implemented from scratch (Multi-head attention, positional encoding, etc.)
- **Weight loading** from pretrained GPT-2 to accelerate training
- **Fine-tuning** using Alpaca-style instruction dataset for better instruction following
- **Interactive Web UI** to test model responses on custom user inputs
- **Example outputs** from the model

---

## 📷 Example Outputs

### Example 1
![Example 1](images/Screenshot (69).png)

### Example 2
![Example 2](images/Screenshot%20(70).png)

### Example 3
![Example 3](images/Screenshot%20(72).png)


## 🏃 How to Run the Project

Follow these steps to set up and run the project on your local machine.

---

### 1️⃣ Clone the Repository

git clone https://github.com/souravsharma22/PA_by_Transformers.git

### 2️⃣ Create the virtual environment and install all dependencies
mentioned in requirements.txt

### 3️⃣ Run the preparingdataset.ipynb file 
load the download the pretrainde weights from openAI
then train the model with your own data
run for one epoch and then save the trained weights as it may take very much time for training
reload the weigths and train again for few epoch 
check the training and validation loss

### 4️⃣ Run app.py
use command - python app.py 

## 👨‍💻 Author

**Sourav Sharma**  
Machine Learning Enthusiast | AI Developer  

📧 Email: souravsharma2210bgp@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sourav-sharma-12b589297/) | [GitHub](https://github.com/souravsharma22)

