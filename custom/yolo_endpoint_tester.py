import requests

url = 'http://labmaite.local:5555/health_check'
files = {'image': open('./lm_analysis_test_img.png', 'rb')}
response = requests.post(url, files=files)

print(response.json())
