from flask import Flask, render_template,session
import os
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import numpy as np
import time
import cv2
import math



UPLOAD_FOLDER = 'F:/social2/image/'
ALLOWED_EXTENSIONS = {'jpg','jpeg','png'}

####
labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "./yolov3.weights"
configPath = "./yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
####

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#@app.route("/")
#def home():
#    return render_template("home.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
@app.route('/result/<filename>')
def uploaded_file(filename):
    imgdir="F:/social2/image/"+filename
    image =cv2.imread(imgdir)
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Frame Prediction Time : {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
            
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    ind = []
    for i in range(0,len(classIDs)):
        if(classIDs[i]==0):
            ind.append(i)
    a = []
    b = []
    color = (0,255,0) 
    if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                a.append(x)
                b.append(y)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            

    distance=[]     
    nsd = []
    for i in range(0,len(a)-1):
        for k in range(1,len(a)):
            if(k==i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                if(d<=75.0):
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))
    print(distance)
    print(nsd)
    color = (0, 0, 255)
    text=""
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "Alert"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
           
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
#cv2.imshow("Social Distancing Detector", image)
    outputfile="output.jpg"
    out="F:/social2/image/output/"+outputfile
    cv2.imwrite(out, image)
    return send_from_directory("F:/social2/image/output/",outputfile)
    
if __name__ == "__main__":
    app.run(debug=True)
