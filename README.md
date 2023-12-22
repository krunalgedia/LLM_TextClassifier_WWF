## LLM_TextClassifier_SwissWWF

Business Problem:

WWF regularly publishes articles, which may sometimes exhibit bias from the writer's perspective. The ideal goal for each article is to be perceived in society as constructive information. However, achieving this requires additional human input for feedback, and even this feedback process can introduce bias. Moreover, given the need to publish articles promptly, relying solely on human feedback may result in delays. Switzerland's WWF aims to address this challenge by automating the feedback process.

Project Objective:

The primary objective of this project is to leverage existing pre-trained Large Language Models (LLM), conduct advanced prompt engineering, and develop a custom web application. This application allows users to upload or write text, receiving immediate feedback on the constructiveness of the content. The system will provide insights into why the text is classified in a specific category. This initiative aims to streamline the article review process, reduce delays, and ensure a more objective evaluation of the content's constructiveness. 

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Workflow](#workflow)
- [Results](#results)
- [More ideas](#More-ideas)
- [Dependencies](#dependencies)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Project Overview

The goal of this project is to develop a real-time web application to
* Classify the text as constructive or not and also to give reasons.

## Installation

```bash
# Example installation command
pip install -r requirements.txt

# Run Web Application
streamlit run app.py
```
Note: if there are any version clashes while installing requirement.txt, please try removing the version number corresponding to the library form requirement.txt and try installing it again.
Also check Dependencies Section for major dependencies and their version. Ensure that atleast those libraries are installed with corresponding version.

## Data

The data used for promt engineering were the key points distinguishing [constructuve and non-constructuve text] (https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/data/processed/Medienmitteilungen%20Export%20DE%2020230822-%20Kriterien%20der%20Konstruktivit%C3%A4t%20updated.csv). 

Further, all the reasons for a text classified as constructive and not constructive present in this [file] (https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/data/raw/Medienmitteilungen%20Export%20DE%2020230822.xlsx) was also used. It consist of 35 constructive text reasons and 15 non-constructive text reason.

The final promt was engineered for OpenAI GPT model. 

## Workflow
The user is only expected to input the text as shown in the sample GIF below:
![image](https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/images/sample%20(1).gif)

The user can further choose whether to have output in Text, JSON or Table form from the left sidebar:
![image](https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/images/sample%20(2).gif)

The user can also change the GPT model to use (given newer and newer models keep comming and not-so-old models keep getting depreciating soon). For this, just enter the GPT model on left sidebar and press enter.

The randomness or temperature parameter on the left sidebar is actually

0. Prepare training data by annotation using the UBIAI tool [1]. This includes drawing bounding boxes and labeling in the BIOES tagging form [2].
1. Importing data
2. Observing data
3. Loading data in appropriate form.
4. Fine-tuning LayoutML model.
5. Preparing test set processing, including OCR of prediction documents using Pytesseract and getting the bounding box for all text in the test sample.
6. Running predictions on the bounding boxes of Pytesseract.
7. Update the database with relevant NER extracted from the model prediction on the annotated test sample.

* notebooks/SBB_TrainTicketParser.ipynb contains the end-to-end code for Document parsing with database integration.
* app.py contains the streamlit app code.

## Results

We fine-tuned using Facebook/Meta's LayoutLM (which utilizes BERT as the backbone and adds two new input embeddings: 2-D position embedding and image embedding) [3]. The model was imported from the Hugging Face library [4] with end-to-end code implemented in PyTorch. We leveraged the tokenizer provided by the library itself. For the test case, we perform the OCR using Pytesseract.

With just 4 SBB train tickets we can achieve an average F1 score of 0.81.   

| Epoch | Average Precision | Average Recall | Average F1 | Average Accuracy |
|--------:|------------:|---------:|-----:|-----------:|
|     145 |        0.89 |     0.77 | 0.82 |       0.9  |
|     146 |        0.9  |     0.79 | 0.84 |       0.9  |
|     147 |        0.86 |     0.77 | 0.81 |       0.89 |
|     148 |        0.87 |     0.78 | 0.82 |       0.9  |
|     149 |        0.86 |     0.77 | 0.81 |       0.89 |

The web application serves demo:
![Image 1](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/sample.gif) | ![Image 2](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/test1.gif)
--- | --- 
Opening page | Testing ... 

Once the user uploads the image, the document gets parsed and the information from the document gets updated in the relational database which can be used to verify the traveler's info and also to automate the travel cost-processing task.


## More ideas

Instead of using OCR from the UBIAI tool, it best is to use pyteserract or same OCR tool for train and test set. Further, with Document AI being developed at a rapid pace, it would be worthwhile to test newer multimodal models which hopefully either provide a new solution for not using OCR or inbuilt OCR since it is important to be consistent in preprocessing train and test set for best results.

Also, train on at least >50 tickets, since this was just a small test case to see how well the model can work.

## Dependencies

This project uses the following dependencies:

- **Conda:** 4.14.0 
- **Python:** 3.10.13
- **OpenAI:** 0.28.1
- **Streamlit:** 1.28.2
- **Pandas:** 2.0.3
 
 Else, full list of dependencies can be found in the [requirement.txt](https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/requirements.txt)
  
## Contact

Feel free to reach out if you have any questions, suggestions, or feedback related to this project. I'd love to hear from you!

- **LinkedIn:** [Krunal Gedia](https://www.linkedin.com/in/krunal-gedia-00188899/)

## References
[1]: UBIAI annotation tool: [UBIAI](https://app.ubiai.tools/Projects).

[2]: Named Entity Recognition, Vijay Krishnan and Vignesh Ganapathy: [NER](http://cs229.stanford.edu/proj2005/KrishnanGanapathy-NamedEntityRecognition.pdf) 

[3]: LayoutLM: Pre-training of Text and Layout for Document Image Understanding: [LayoutLM] (https://arxiv.org/abs/1912.13318)

[4]: [Huggingface LayoutLM] (https://huggingface.co/docs/transformers/model_doc/layoutlm)


