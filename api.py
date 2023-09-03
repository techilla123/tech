import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from flask import Flask, request, Response, jsonify
from base64 import encodebytes
import base64
import io
from PIL import Image
from flask_cors import CORS
import requests

app= Flask(__name__)
CORS(app)
fapp=FaceAnalysis(name='buffalo_l')
fapp.prepare(ctx_id=0,det_size=(640,640))
swapper=insightface.model_zoo.get_model('inswapper_128.onnx',download=False,download_zip=False)

paths={"1":"1.jpg","2":"2.jpeg","3":"3.jpg","4":"4.jpeg","5":"5.jpeg", "6":"6.jpeg","7":"7.png","8":"8.jpg"}
@app.route('/rec', methods=['POST'])
def route():
    try:
        
        r = request.get_json()
        
        nparr = np.fromstring(base64.b64decode(r['image']), np.uint8)
    # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        choice=r['choice']
        cv2.imwrite("tom.jpg", img)
        main_face=fapp.get(img)
        main_face=main_face[0]
        poster=cv2.imread('./images/final/'+paths[choice.strip()])
        face=fapp.get(poster)
        res=poster.copy()
        for faces in face:
            res=swapper.get(res,faces,main_face,paste_back=True)
        cv2.imwrite("niru.jpg", res)
        print(type(res))

        
        

    #nparr = np.fromstring(base64.b64decode(data, np.uint8))
    # decode image
    #img1 = cv2.imdecode(data, cv2.IMREAD_COLOR)
    #cv2.imwrite("tom.jpg", img1)
        string = base64.b64encode(cv2.imencode('.jpg', res)[1]).decode()
        #response=jsonify({'result': string,'errorcode':'0'})
        
        #response.headers.add('Access-Control-Allow-Origin', '*')
    
        #return response
        return jsonify({'result': string,'errorcode':'0'})
    except:
        #response=jsonify({'result': '','errorcode':'1'})
        #response.headers.add('Access-Control-Allow-Origin', '*')
        
        #return response
        return jsonify({'result': '','errorcode':'1'})


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img
#girl=cv2.imread('yashi.jpeg')
#girl_faces=app.get(girl)
#girl_faces=girl_faces[0]
#male=cv2.imread('super.png')
#face=app.get(male)
#res=[]
#girl_faces=app.get(girl)
#girl_face=girl_faces[0]
#res=male.copy()
#for faces in face:
#   res=swapper.get(res,faces,girl_face,paste_back=True)
#cv2.imwrite("niru.jpg", res)
app.run(host="0.0.0.0", port=5000)