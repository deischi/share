import onnx
import onnxruntime as rt
import joblib
import cv2
import numpy as np  
device = 'cuda:0' #'cuda:0' 

if __name__ == "__main__":  		
    us_onnx_model="mask-rcnn_epoch_5.onnx"
    fib_execution_provider=['DmlExecutionProvider']  #['DmlExecutionProvider']
    with open(us_onnx_model, "rb") as f:		
         sess_mrcnn = rt.InferenceSession(f.read(),providers=fib_execution_provider)		
         
    input_name_mrcnn = sess_mrcnn.get_inputs()[0].name
    img = cv2.imread("testImage.PNG",0)
    image = img
    image = np.expand_dims(image, axis=0)	 
    image = np.expand_dims(image, axis=0)	 
    image = image/image.max()		 
    pred  = sess_mrcnn.run(None, {input_name_mrcnn: image.astype(np.float32)})			
    box   =  pred[0]
    score = pred[2]
    mask  = pred[3]			
            
    print(box, score)
    for i,curr_mask in enumerate(mask):
        temp  = curr_mask[0]
        temp[temp>0.1]=1
        temp[temp<0.1]=0
        if np.sum(temp)>0:
            cv2.imwrite('mask'+str(i+1)+'.png', (255*temp))



	 