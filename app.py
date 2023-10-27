# index 화면에서 파일 선택 후 예측 버튼을 클릭하면
# 어떤 이미지 인지 분류하여 화면에 출력
import pandas as pd
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# 모델 불러오기
model = tf.keras.models.load_model('improved_food_classification_model_updated.h5')
# CSV 파일 읽어오기

df=pd.read_csv('nutrition101_K.csv', encoding='CP949')
labels = df['name'].tolist()

labels = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	# 전송 받은 파일이 이미지 파일 형태가 아니면 400을 반환
	if 'image' not in request.files:
		return jsonify({'error': 'no file upload'}), 400

	f = request.files['image']
	if f.filename == '':
		return jsonify({'error': 'no file selected'}), 400

	image = Image.open(f)
	image = image.resize((299, 299))

	image = image.convert('RGB')

	# 이미지를 numpy 배열로 변환
	image_array = np.array(image)
	image_array=image_array/255.0
	# 모델에 이미지 데이터 전달
	image_array = image_array.reshape(1, *image_array.shape)  # 배치 차원 추가
	#이미지예측
	pred = model.predict(image_array)
	pred_class = np.argmax(pred, axis=-1)[0]
	prediction_label = labels[pred_class]
	print(df[df['name'] == prediction_label].to_dict('records'))
	# 예측된 음식에 해당하는 정보 찾기
	#food_info = df[df['name'] == prediction_label].to_dict('records')
	food_info = df[df['name'] == prediction_label].to_dict('records')
	print(food_info)
	return jsonify({'prediction': prediction_label,'food_info':food_info})

if __name__ == '__main__':
    app.run(debug=True)


