import json
import boto3
import botocore.config

from datetime import datetime

# Main aim is to create a blog. Hence, creating a Lambda function
def blog_generate_using_bedrock(BlogTopic:str) -> str:
    """
    This function will generate a blog using bedrock
    :param BlogTopic: Topic of the blog
    :return: Blog URL
    """
    
    prompt = """
        <s>[INST]Human: Write a 200 word blog on the topic {BlogTopic}
        Assistant:[/INST]
    """
    
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    
    try:
        bedrock = boto3.client(
            "bedrock-runtime", 
            region_name="us-east-1", 
            config=botocore.config.Config(read_timeout=300, retries={"max_attempts": 3})
        )
        
        response = bedrock.invoke_model(
            body=json.dumps(body), 
            modelId="meta.llama3-8b-instruct-v1:0"
        )
        
        response_content = response.get('body').read()
        
        response_data = json.loads(response_content)
        
        print(response_data)
        
        blog_details = response_data['generation']
        
        return blog_details
    
    except Exception as e:
        print(f"Error Generating the blog: {e}")
        return ""

def save_blog_details_s3(s3_key: str, s3_bucket:str, generate_blog: str) -> None:
    """
    This function will save the blog details to S3
    :param s3_key: S3 key
    :param s3_bucket: S3 bucket
    :return: None
    """
    s3 = boto3.client('s3')
    
    try:
        s3.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=generate_blog
        )
        print("Blog saved to s3")
        
    except Exception as e:
        print(f"Error saving the blog to s3: {e}")

def lambda_handler(event, context):
    """
    This function will be called by the lambda function
    :param event: Event object
    :param context: Context object
    :return: None
    """
    
    event = json.loads(event['body'])
    BlogTopic=event['blog_topic']
    
    generate_blog = blog_generate_using_bedrock(BlogTopic=BlogTopic)
    
    if generate_blog:
        current_time = datetime.now().strftime("%H%M%S")
        s3_key = f"blog-output/{current_time}.txt"
        s3_bucket = "aws_bedrock_course1"
        
        save_blog_details_s3(
            s3_key=s3_key,
            s3_bucket=s3_bucket,
            generate_blog=generate_blog
        )
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Blog Generated Successfully",
                "data": generate_blog
                })
            }
    else:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "Error Generating the Blog"
                })
            }
        
    