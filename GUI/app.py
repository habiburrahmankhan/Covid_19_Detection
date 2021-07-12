from flask import Flask , render_template , request
from flask_ngrok import run_with_ngrok 
import utils 

from PIL import Image
import gradcam
app = Flask(__name__)
run_with_ngrok(app)

@app.route("/")
def indexpage():
    return render_template("index.html") 

@app.route("/about")
def aboutt():
    return render_template("about.html")

@app.route("/submit" , methods = ["POST"])
def submitdata():
    if request.method == "POST":
        f = request.files["userfile"]
        print(f , "fffff")
        
        path = "/content/gdrive/MyDrive/minor_project_new/GUI/static/{}".format(f.filename)
        f.save(path)
        print(path)
        l , p   = utils.predict(path)   #("COVID",[0.98,0.02])  
        grad_img = gradcam.gradcam_image_process(path , l , p )
#         j = 0
#         for i in grad_img:
     
        img = Image.fromarray(grad_img)
#             j+=1
        print(img) 
        img.save("/content/gdrive/MyDrive/minor_project_new/GUI/static/inception_img.jpeg")  
        my_ans = {
             "label" : l , 
             "image" :  "./static/{}".format(f.filename),
            
             "imgincep" :   "/content/gdrive/MyDrive/minor_project_new/GUI/static/inception_img.jpeg",
#             "imagedesnet" :  "./static/gradcam{}".format(utils.models_name[1] + "img"),
#             "imageresnet" :  "./static/gradcam{}".format(utils.models_name[2] + "img"),
            "neg" : p[1] , 
            "pos" : p[0]
             }
       
        print("Analysis done")   
    return render_template("index.html" ,  my_res = my_ans )
    
# debug=True , use_reloader=False
    
if __name__ == "__main__":
    app.run()