import requests
import os
import time

def send_img(img):
    file_path = img
    target_url = 'http://127.0.0.1:5000/predict'  # 타겟 주소

    with open(file_path, 'rb') as f:
        files = {'file': f}
        res = requests.post(target_url, files=files)

    if res.status_code == 200:
        res = res.json()
        print(res)
        anomaly_score = res.get('prediction')
        instance_score_plot_path = res.get('instance_score_plot_path')
        feature_outlier_image_path = res.get('feature_outlier_image_path')

        # Anomaly Score x이상시 이벤트 처리 구현
        if float(anomaly_score.split(': ')[1]) >= 0.005:
            print('이상 감지')
        else:
            print('이상 없음')

        # Save the images if paths are provided
        if instance_score_plot_path:
            save_image(instance_score_plot_path, 'instance_score_plot')

        if feature_outlier_image_path:
            save_image(feature_outlier_image_path, 'feature_outlier_image')
        
    else:
        print('error:', res.text)

def save_image(url, folder_prefix):
    if not os.path.exists("results"):
        os.makedirs("results")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"results/{folder_prefix}_{int(time.time())}.png", 'wb') as f:
                f.write(response.content)
            print(f"Saved {folder_prefix} image successfully")
        else:
            print(f"Failed to fetch {folder_prefix} image: {response.status_code}")
    except Exception as e:
        print(f"Error saving {folder_prefix} image: {e}")

# 단일 이미지 테스트
if __name__ == '__main__':
    file_path = 'Positive/00062.jpg'  # 보낼 이미지 파일 경로
    send_img(file_path)
