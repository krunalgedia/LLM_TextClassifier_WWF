FROM python:3.9.19

WORKDIR /app

RUN pip install --upgrade pip
#COPY requirements.txt /app/

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pandas
RUN pip install openai==0.28.1
RUN pip install python-dotenv
RUN pip install altair==4.0
RUN pip install urllib3==1.26.6
RUN pip install streamlit==1.24.0
RUN pip install -U scikit-learn
RUN pip install openpyxl

COPY . /app

EXPOSE 8501
#EXPOSE 8080

#CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#CMD ["streamlit", "run", "app.py"]
#CMD ["streamlit", "run", "--server.address", "0.0.0.0", "app.py"]
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0","--server.enableCORS=false","--server.enableWebsocketCompression=false"]
