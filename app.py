import os
from flask import Flask, render_template, request
from img2gray import segmented, pca
import inferencecopy2
from PIL import Image
import Final

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html", value="show")


@app.route("/upload", methods=["post", "get"])
def upload():
            
    target = os.path.join(APP_ROOT, "static/")
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "".join([target, filename])
        print(destination)
        #os.remove(r".\static\segmented.jpg")
        file.save(destination)
        
        inferencecopy2.DeeplabSeg(Image.open(destination))
        classpredic = Final.LdaAnalysis(Image.open(r'.\static\segmented.jpg'))
   
        #os.remove("segmented.jpg")

    return render_template("complete.html",image_name1='segmented.jpg', image_name2=filename, value = classpredic)


if __name__ == "__main__":
    app.run(port=5250, debug=True)






