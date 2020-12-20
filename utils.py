import numpy as np
import logging
import math
import boto3
from PIL import Image
import cv2


def importImage(im, ratio):
    logging.debug("Converting image to Numpy Array")
    im = np.asarray(Image.open(im))
            
    logging.debug("Scaling and removing black levels")
    im = np.maximum(im - 0.0, 0) / (255.0 - 0.0)  # subtract the black level

    logging.debug("Cast to Float32")
    im = im.astype(np.float32)

    im *= ratio    
    return im

def inferTransform(im):
    # scaling image down to a max dimension of 512, maintaining aspect ratio
    logging.info("Imported image is: %d X %d" % (im.shape[0], im.shape[1]))
    if max(im.shape) > 512:
        scale_factor = 512 / max(im.shape)
        H = int(im.shape[0] * scale_factor)
        W = int(im.shape[1] * scale_factor)
        logging.info("Rescaling image to: %d X %d" % (H, W))
        im = cv2.resize(im, (W,H), cv2.INTER_AREA)

    # cropping image to nearest 16, to allow torch to compute
    logging.debug("Trimming image to size")
    H = math.floor(im.shape[0]/16.0)*16
    W = math.floor(im.shape[1]/16.0)*16
    im = im[:H, :W, :]

    return im

def read_queue(queue_url):
    # Receive message from SQS queue
    sqs = boto3.client('sqs')
    response = sqs.receive_message(
        QueueUrl=queue_url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=0,
        WaitTimeSeconds=0
    )
    try:
        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']
        return [message, receipt_handle]
    except:
        return [-1, -1]

def terminate_instance():
    client = boto3.client('ec2')
    response = client.describe_instances(
    Filters=[
        {
            'Name': 'subnet-id',
            'Values': ['subnet-0d6e5384d0fb5b377']
        },
    ],
    MaxResults=5
    )
    
    instance = response['Reservations'][0]['Instances'][0]['InstanceId']
    
    client.terminate_instances(
    InstanceIds=[
        instance
    ],)

    