# Generative AI based Image Caption Generator

![Streamlit](https://img.shields.io/badge/Platform-Streamlit-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)

An interactive web application that generates captions for images using state-of-the-art AI models. Upload an image, choose from multiple captioning features, and explore translations and stylistic variations in real-time.

---

## 🔍 Features

* **Multiple Captions**: Generate 1–5 diverse captions using beam search or sampling strategies.
* **Caption Styles**: Transform a base caption into descriptive, funny or poetic variations.
* **Translate Caption**: Translate the generated caption to Hindi or Spanish(more translators coming soon...).

---

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ankit-1607/Gen-AI-based-Image-Caption-Generator.git
   cd Gen-AI-based-Image-Caption-Generator
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Run Locally

```bash
streamlit run app.py
```

### 2. Via Google Colab

> *Note: Running the application via colab will give better performance than Streamlit deployed app due to free tier resource allocation difference.*

> **How to run the application using google colab:**

* Change the runtime type to ‘T4 GPU’ to observe significantly better response time after intital model load.

* ‘Run All’ cells.

* Scroll to the last cell which uses localtunnel to provide a temporary online url to view the application.

* Copy the IP address of ‘External URL’.

*  e.g. in the below shown image copy 34.16.168.188

![image](https://github.com/user-attachments/assets/528e4434-64cc-4436-8f8b-203aa595132f)

* Click on the ‘your url’ and go to the page link and paste the copied IP Address as tunnel password and explore the application.

**Google Colab link:** [Open in Colab](https://colab.research.google.com/drive/1v-NeTNbOsabLxek-pihdkxiAI1vRfI18?usp=sharing)

---

### 3. Deployed Streamlit App

Access the live application deployed on Streamlit sharing:

**Streamlit Deployed Application**:[AI Image Caption Generator](https://gen-ai-image-caption-generator.streamlit.app/)

> *Note: The Streamlit deployment is hosted on a free tier, so performance may be slower compared to running on a GPU-enabled Colab instance.*

---

### 📊 Sample Outputs
> Welcome Page.

![image](https://github.com/user-attachments/assets/ad573b59-844f-4110-81e7-679efdfdbdf1)

> Multiple Captions generated using Beam/Greedy Search.

![image](https://github.com/user-attachments/assets/e2ad34dd-eff3-4065-b598-540fa62af854)

> Multiple Captions generated using Sampling Search(generates diverse outputs).

![image](https://github.com/user-attachments/assets/e1bfe4da-4474-408b-937e-b4f04aa294ea)

> A descriptive caption generated using caption styling.

![image](https://github.com/user-attachments/assets/474e6e4e-6561-4163-9b4f-1efb6a9ed5df)

> A funny caption generated using caption styling.

![image](https://github.com/user-attachments/assets/c590d06d-1aca-4414-814c-4072cf48148f)

> A poetic caption generated using caption styling.

![image](https://github.com/user-attachments/assets/f4301c8c-4aa9-4df9-a0ed-689da72c2edf)

> Spanish Caption generated using translate caption.

![image](https://github.com/user-attachments/assets/578862b6-192a-4895-a471-80bec3b87233)


---

## 📄 Requirements

* Python 3.10
* streamlit
* torch
* transformers
* pillow
* sentencepiece

Install via:

```bash
pip install -r requirements.txt
```

---

## ❤️ Acknowledgements

* [Salesforce BLIP](https://github.com/salesforce/BLIP) for image captioning models.
* [LaMini-Flan-T5](https://huggingface.co/MBZUAI/LaMini-Flan-T5-783M) for text transformation.
* [Hugging Face Transformers](https://huggingface.co/transformers/) library.
* [Streamlit](https://streamlit.io/) for the web interface.
