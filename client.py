import requests

file_path = 'Positive/00062.jpg' # 보낼 이미지 파일 경로

target_url = 'http://127.0.0.1:5000/predict' # 타겟 주소

with open(file_path, 'rb') as f:
    files = {'file' : f}
    res = requests.post(target_url, files=files)

if res.status_code == 200:
    res = res.json()
    print(res)
    # Anomaly Score x이상시 이벤트 처리 구현 Todo
    anomaly_score = res['prediction']
    if float(anomaly_score.split(': ')[1]) >= 0.005:
        print('이상 감지')
    else :
        print('이상 없음')

else :
    print('error :', res.text)
