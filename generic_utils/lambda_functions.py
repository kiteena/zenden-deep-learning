#upload
import json
import base64
import boto3

BUCKET_NAME = '<bucketname>'

def lambda_handler(event, context):
    file_name = event['filename']
    file_content = event['content']
    s3 = boto3.client('s3')
    try:
        s3_response = s3.put_object(Bucket=BUCKET_NAME, Key=file_name, Body=file_content)
    except Exception as e:
        raise IOError(e)
    return {
        'statusCode': 200,
        'body': {
            'file_name': file_name
        }
    }     

#delete
import json
import base64
import boto3

BUCKET_NAME = '<bucketname>'

def lambda_handler(event, context):
    file_name = event['filename']
    s3 = boto3.client('s3')
    try:
        s3_response = s3.delete_object(Bucket=BUCKET_NAME, Key=file_name)
    except Exception as e:
        raise IOError(e)
    return {
        'statusCode': 200,
        'body': {
            'filename': file_name
        }
    }     
