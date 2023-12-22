## LLM_TextClassifier_SwissWWF

Business Problem:

WWF regularly publishes articles, which may sometimes exhibit bias from the writer's perspective. The ideal goal for each article is to be perceived in society as constructive information. However, achieving this requires additional human input for feedback, and even this feedback process can introduce bias. Moreover, given the need to publish articles promptly, relying solely on human feedback may result in delays. Switzerland's WWF aims to address this challenge by automating the feedback process.

Project Objective:

The primary objective of this project is to leverage existing pre-trained Large Language Models (LLM), conduct advanced prompt engineering, and develop a custom web application. This application allows users to upload or write text, receiving immediate feedback on the constructiveness of the content. The system will provide insights into why the text is classified in a specific category. This initiative aims to streamline the article review process, reduce delays, and ensure a more objective evaluation of the content's constructiveness. 

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data](#data)
- [Workflow](#workflow)
- [Major files](#major-files)
- [Dependencies](#dependencies)
- [Contact](#contact)
- [Team](#team)

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
Note: if there are any version clashes while installing requirement.txt, please try removing the version number corresponding to the library form requirement.txt and installing it again.
Also, check **Dependencies Section** for major dependencies and their version. Ensure that at least those libraries are installed with the corresponding version.

## Data

The data used for prompt engineering were the key points distinguishing [constructuve and non-constructuve text] (https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/data/processed/Medienmitteilungen%20Export%20DE%2020230822-%20Kriterien%20der%20Konstruktivit%C3%A4t%20updated.csv). 

Further, all the reasons for a text classified as constructive and not constructive are present in this [file] (https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/data/raw/Medienmitteilungen%20Export%20DE%2020230822.xlsx) was also used. It consists of 35 constructive text reasons and 15 non-constructive text reasons.

The final prompt was engineered for the OpenAI GPT model. 

## Workflow
The user must export their OpenAI API key either as an environment variable. For this, just replace the key in [.env file] (https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/.env) with your key.

After performing the above step,

* The user is only expected to input the text as shown in the sample GIF below:
![image](https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/images/sample%20(1).gif)


* The user can further choose whether to have output in Text, JSON or Table form from the left sidebar:
![image](https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/images/sample%20(2).gif)


* The user can also change the GPT model to use (given newer and newer models keep coming and not-so-old models keep depreciating soon). For this, just enter the GPT model on the left sidebar and press enter.

* The randomness or temperature parameter on the left sidebar is a model parameter on the level of randomness of the GPT model output. The normal chatbot available on the ChatGPT website has a 0.5 value. For more definitive output like the one required for our goal, it is recommended by OpenAI to keep it at 0.2, especially for GPT 3.5 models. Thus, by default, the value of this parameter is 0.2 on the left sidebar. 

## Major Files

* .env contains the OpenAI API key.
* app.py contains the streamlit app code.
* data folder contains data used by the app's built-in prompt.

## Dependencies

This project uses the following dependencies:

- **Conda:** 4.14.0 
- **Python:** 3.10.13
- **OpenAI:** 0.28.1
- **Streamlit:** 1.28.2
- **Pandas:** 2.0.3
- **dotenv**: 1.0.0

```
conda install conda=4.14.0

conda create -n your_environment_name python=3.10.13
conda activate your_environment_name

pip install openai==0.28.1

pip install streamlit==1.28.2

pip install pandas==2.0.3

pip install python-dotenv==1.0.0
```

 Else, the full list of dependencies can be found in the [requirement.txt](https://github.com/krunalgedia/LLM_TextClassifier_WWF/blob/main/requirements.txt)
  
## Contact

Feel free to reach out if you have any questions, suggestions, or feedback related to this project. I'd love to hear from you!

- **LinkedIn:** [Krunal Gedia](https://www.linkedin.com/in/krunal-gedia-00188899/)

## Team

Szymon Rucinski, Philipp Pestlin, Krunal Gedia
