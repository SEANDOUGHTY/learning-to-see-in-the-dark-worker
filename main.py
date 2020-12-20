import torch
from utils import *
from src.model.model import UNet
import numpy as np
from PIL import Image
import boto3
import io
import os
import time
from logging.config import dictConfig
import logger_config

dictConfig(logger_config.CONFIG)

QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/195691282245/learningtoseeinthedark'
TIMEOUT = 600
bucketName = 'pyseedarkresources'
checkpoint_path = "checkpoint/checkpoint.t7"

# Starting AWS Session
logging.info("Creating AWS Session")
try:
    session = boto3.Session()
except:
    logging.error("AWS Session Failed: Check Access Key and Permissions")
    quit()

# Getting checkpoint file
s3 = boto3.client('s3')
logging.info("Downloading checkpoint file: %s from S3" % checkpoint_path)
s3.download_file(bucketName, checkpoint_path, checkpoint_path)

# Setting Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
logging.info("Set torch device as: %s" % device)

# Load model
logging.info ("Loading UNet Model to %s"% device)
try:
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
except:
    logging.error("Model Unsuccessfully Loaded")
    quit()

logging.info("Model Successfully Loaded")
#set model to evaluate mode
model.eval()

if __name__ == '__main__':
    # Create SQS client 
    last_job = time.time()
    while True:
        [message, receipt_handle] = read_queue(QUEUE_URL)
        if message == -1:
            ellapsed_seconds = time.time() - last_job
            logging.info('Queue is empty %d seconds until shutdown' % int(TIMEOUT-ellapsed_seconds))
            if ellapsed_seconds > TIMEOUT:
                break
            else:
                time.sleep(5)
                continue

        [fileName, ratio] = message['Body'].split(', ')       
        logging.info("Recieved message to process: %s with ratio: %s" % (fileName, ratio))

        ratio = float(ratio)
        inputImage = "inputs/" + fileName
        outputImage = "outputs/" + 'output_' + fileName[6:]
        
        logging.info("Downloading image from S3")
        try:
            s3 = session.resource('s3')
            object = s3.Object(bucketName, inputImage)
            
            image = io.BytesIO()
            object.download_fileobj(image)
            image.seek(0)
        except:
            logging.error("Unable to Download Input Image from S3")
            
        logging.info("Importing Image from: %s" % inputImage)
        im = importImage(image, ratio)
        
        logging.info("Transforming Image")
        im = inferTransform(im)
        
        logging.info("Loading Input Image to %s" % device)
        tensor = torch.from_numpy(im).transpose(0, 2).unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():    
            # Inference
            logging.info("Performing Inference on Input Image")
            try:
                output = model(tensor)

                # Post processing for RGB output
                output = output.to('cpu').numpy() * 255
                output = output.squeeze()
                output = np.transpose(output, (2, 1, 0)).astype('uint8')
                output = Image.fromarray(output).convert("RGB")
                #output.show()

                # Output buffer for upload to S3
                buffer = io.BytesIO()            
                output.save(buffer, "PNG")
                buffer.seek(0) # rewind pointer back to start

                logging.info("Uploading Image to %s", outputImage)
                try:
                    s3.Bucket(bucketName).put_object(
                        Key=outputImage,
                        Body=buffer,
                        ContentType='image/png',
                    )
                except:
                    logging.error("Failed to Upload Image")
            except:
                logging.error("Inference Failed")  

        # Delete received message from queue
        sqs = boto3.client('sqs',
            region_name = 'us-east-1')
        sqs.delete_message(
            QueueUrl=QUEUE_URL,
            ReceiptHandle=receipt_handle
        )

        print('Received and deleted message: %s' % message)
        last_job = time.time()
    
    logging.info("Terminating Instance")
    terminate_instance()

    