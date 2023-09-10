import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from flask import Flask, request, Response, jsonify

import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer


from base64 import encodebytes
import base64
import io
from PIL import Image
from flask_cors import CORS
app= Flask(__name__)
CORS(app)
fapp=FaceAnalysis(name='buffalo_l')
fapp.prepare(ctx_id=0,det_size=(640,640))
swapper=insightface.model_zoo.get_model('./models/inswapper_128.onnx',download=False,download_zip=False)

paths={"1":"1.jpg","2":"2.jpeg","3":"3.jpg","4":"4.jpeg","5":"5.jpeg", "6":"6.jpg","7":"7.jpeg","8":"8.jpg","9":"9.jpg","10":"10.jpg"}

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = './models/realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
print(half)
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

os.makedirs('output', exist_ok=True)


@app.route('/rec', methods=['POST'])
def route():
    try:
        
        r = request.get_json()
        
        nparr = np.fromstring(base64.b64decode(r['image']), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        main_face=fapp.get(img)
        main_face=main_face[0]
        #cv2.imwrite("tom.jpg", img)
        #choice=r['choice']
        nparr1 = np.fromstring(base64.b64decode(r['choice']), np.uint8)
        poster= cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        face=fapp.get(poster)
        #poster=cv2.imread('./images/final/'+paths[choice.strip()])
        
        res=poster.copy()
        for faces in face:
            res=swapper.get(res,faces,main_face,paste_back=True)
        cv2.imwrite("niru.jpg", res)
        output=inference("niru.jpg", 'v1.4', 2)
    
    #nparr = np.fromstring(base64.b64decode(data, np.uint8))
    # decode image
    #img1 = cv2.imdecode(data, cv2.IMREAD_COLOR)
    #cv2.imwrite("tom.jpg", img1)
        string = base64.b64encode(cv2.imencode('.jpg', output)[1]).decode()
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


def inference(img, version, scale):
    # weight /= 100
    print(img, version, scale)
    try:
        #extension = os.path.splitext(os.path.basename(str(img)))[1]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        if version == 'v1.2':
            face_enhancer = GFPGANer(
            model_path='./models/GFPGANv1.2.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'v1.3':
            face_enhancer = GFPGANer(
            model_path='./models/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'v1.4':
            face_enhancer = GFPGANer(
            model_path='./models/GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'RestoreFormer':
            face_enhancer = GFPGANer(
            model_path='./models/RestoreFormer.pth', upscale=2, arch='RestoreFormer', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'CodeFormer':
             face_enhancer = GFPGANer(
             model_path='./models/CodeFormer.pth', upscale=2, arch='CodeFormer', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'RealESR-General-x4v3':
             face_enhancer = GFPGANer(
             model_path='./models/realesr-general-x4v3.pth', upscale=2, arch='realesr-general', channel_multiplier=2, bg_upsampler=upsampler)

        try:
            # _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as error:
            print('Error', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('wrong scale input.', error)
       
        #cv2.imwrite(save_path, output)

        #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output
    except Exception as error:
        print('global exception', error)
        return None, None



app.run(host="0.0.0.0", port=5000)
