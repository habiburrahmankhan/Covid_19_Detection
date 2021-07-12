import tensorflow as tf;
import utils 
from PIL import Image
import cv2
import numpy as np 


class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    def compute_heatmap(self, image, eps=1e-8):
        gradModel = tf.keras.Model( inputs=[self.model.inputs] ,  outputs=[   self.model.get_layer(self.layerName).output ,  self.model.output ]    )
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
# resize the heatmap to oringnal X-Ray image size
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
# normalize the heatmap


        numer = heatmap
        denom = heatmap.max()
        heatmap = numer / denom
        heatmap = heatmap*255
      
        heatmap =heatmap.astype("uint8")
       
        return heatmap
    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS): # COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        return (heatmap, output)

    
    
def gradcam_image_process(filename , l , p ):
    orignal = cv2.imread(filename)
    orig = cv2.cvtColor(orignal, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(orig, (224, 224))
    dataXG = np.array(resized) / 255.0
    dataXG = np.expand_dims(dataXG, axis=0)
    i = 0 if utils.categories[0]==l else  1 
    new_models = utils.models
    new_model = utils.models[0]
    cam = GradCAM(model=new_model, classIdx=i, layerName='mixed10') # find the last 4d shape "mixed10" in this case
    heatmap = cam.compute_heatmap(dataXG)
    # Old fashioned way to overlay a transparent heatmap onto original image, the same as above
    heatmapY , output = cam.overlay_heatmap(heatmap , resized)
        #lis.append(heatmap)
        #lis.append(heatmapY)
        #lis.append(output)
        #gradimg.append(output)
    # draw the orignal x-ray, the heatmap, and the overlay together
    return output