import cv2
import numpy as np
import pandas as pd
from openvino.inference_engine import IECore
import time
import datetime
import boto3

# Set paramters
ssd_model_xml = 'frozen_inference_graph.xml'
ssd_model_bin = 'frozen_inference_graph.bin'
ssd_device = 'MYRIAD' #'CPU'  #device to run the inference on

cl_model_xml = 'mobilenetv2.xml'
cl_model_bin = 'mobilenetv2.bin'
cl_device = 'MYRIAD' #'CPU'  #device to run the inference on

bird_class_file = 'birdclass.csv'
image_interval = 1  #seconds between frame catpure
image_max = 30      #number of images to capture from video
upload_s3 = False
image_debug = True
count = 1

# Load labels
birdclass = pd.read_csv('birdclass.csv')

# Load the model.
ie = IECore()
ssd_net = ie.read_network(model=ssd_model_xml, weights=ssd_model_bin)
cl_net = ie.read_network(model=cl_model_xml, weights=cl_model_bin)

# Loading model to the plugin 
ssd_exec_net = ie.load_network(network=ssd_net, device_name=ssd_device)
cl_exec_net = ie.load_network(network=cl_net, device_name=cl_device)

# Gather layer names from net.inputs and net.outputs
# The net.inputs object is a dictionary that maps input layer names to DataPtr objects.
ssd_input_blob = next(iter(ssd_net.inputs)) 
ssd_out_blob = next(iter(ssd_net.outputs))  
cl_input_blob = next(iter(cl_net.inputs)) 
cl_out_blob = next(iter(cl_net.outputs))  

# Read network input shape
ssd_n, ssd_c, ssd_h, ssd_w = ssd_net.inputs[ssd_input_blob].shape  
cl_n, cl_c, cl_h, cl_w = cl_net.inputs[cl_input_blob].shape  

# Open the device at the ID 0
cap = cv2.VideoCapture(0)

#Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print('Could not open video device')

#To set the resolution    
cap.set(3, 240)
cap.set(4, 320)


while(True):
    # Capture frame-by-frame
    while count <= image_max:
  
        print ('Process new frame: ', count)
        ret, frame = cap.read()
        
        #pre-process images for model input
        images = np.ndarray(shape=(ssd_n, ssd_c, ssd_h, ssd_w))
        image = frame

        if image.shape[:-1] != (ssd_h, ssd_w):
            #print('Image resized from {} to {}'.format(image.shape[:-1], (ssd_h, ssd_w)))
            image = cv2.resize(image, (ssd_w, ssd_h))          
        
        # Change data layout from HWC to CHW
        images = image.transpose((2, 0, 1))  
    
        # Start inference
        ssd_res = ssd_exec_net.infer(inputs={ssd_input_blob: images})
        ssd_out = np.squeeze(ssd_res[ssd_out_blob])
        
        #Print inference probabilities
        #print('out: ',ssd_out[ssd_out[:,1]!=0])
        bcnt=1
        for x in ssd_out:
            if x[1] == 16:
             
                print('Bird found.', 'Probability:', x[2])

                #draw bounding box
                scale=.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                pt1 = (int(x[3]*ssd_w), int(x[4]*ssd_h))  #(xmin,-ymin)
                pt2 = (int(x[5]*ssd_w), int(x[6]*ssd_h))  #(xmax,-ymax)
                cv2.rectangle(image, pt1, pt2, color=(0, 255, 0),thickness=1)
                
                # Save the bounding box frame to file.
                if image_debug:
                    cv2.imwrite('out\image_box_{}_{}.jpg'.format(count,bcnt), image)
                
                #crop
                crop_img = image[int(x[4]*ssd_h):int(x[6]*ssd_h), int(x[3]*ssd_w):int(x[5]*ssd_w)]
                if image_debug:
                    cv2.imwrite('out\image_crop_{}_{}.jpg'.format(count,bcnt), crop_img)
                
                #pre-process images for model input
                cl_image_in = np.ndarray(shape=(cl_n, cl_c, cl_h, cl_w))
                cl_image = crop_img
                
                if cl_image.shape[:-1] != (cl_h, cl_w):
                    #print('Image resized from {} to {}'.format(image.shape[:-1], (cl_w, cl_h)))
                    cl_image = cv2.resize(cl_image, (cl_w, cl_h))          
                
                # Change data layout from HWC to CHW
                cl_image_in = cl_image.transpose((2, 0, 1)) 
                
                # Start inference
                cl_res = cl_exec_net.infer(inputs={cl_input_blob: cl_image_in})
                cl_out = np.squeeze(cl_res[cl_out_blob])
                cl_predict= np.argmax(cl_out)
                cl_name = birdclass.loc[birdclass['birdclass']==cl_predict,['birdname']].to_string(index=False, header=None)
                print('class:', cl_name, '  Probability:', cl_out[cl_predict])
                
                cv2.putText(image, cl_name, pt1, font, scale, color = (0, 0, 255), thickness=1)
                
                # Prepare file for upload
                dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                file_name = dt+'_'+str(count)+'_'+str(bcnt)+'_'+cl_name.lstrip()+'.jpg'
                print(file_name)
                if upload_s3:
                    image_string = cv2.imencode('.jpg', image)[1].tostring()
                    client = boto3.client('s3')
                    client.put_object(Bucket='r-pi-data', Key = file_name, Body=image_string)
                bcnt +=1
        if image_debug:
            cv2.imwrite('out\image_original_{}.jpg'.format(count), frame)
            cv2.imwrite('out\image_end_{}.jpg'.format(count), image)
        count += 1
        time.sleep(image_interval)
           
    if cv2.waitKey(0): #break out of loop
        break
        
cap.release()

