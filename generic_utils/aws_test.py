import boto3

AccessKeyID= '<insert>'
SecretAccessKey= '<insert>'

s3 = boto3.client ('s3', aws_access_key_id=AccessKeyID, aws_secret_access_key=SecretAccessKey)

# for bucket in s3.buckets.all():
#     print(bucket.name)


# s3.download_file('zenden-cnn','Bryce canyon.jpg','deep_learning/serialized_models/data/Bryce canyon.jpg')
s3.upload_file('/home/kristina/Downloads/house4.jpg', 'zenden-cnn', 'house4.jpg')
