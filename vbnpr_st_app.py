from ultralytics import YOLO
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import tempfile
import streamlit as st
import requests
from bs4 import BeautifulSoup



st.set_page_config(layout="wide", page_title="CarBrand and PlateNumber Detector")


st.title("## Car Brand and Plate Number Detector")
st.write(
    "To identify the brand, colour of your car, get the plate number text (Nigerian though) "
)
st.sidebar.write("## Upload")

st.write("Please upload an image")




#loading the model
car_model = YOLO('yolov8m.pt')
brand_plate_model = YOLO('vbnpr_model.pt')
ocr_model = YOLO("plate_ocr.pt")
colors = pd.read_csv('colors.csv')
url = 'https://nvis.frsc.gov.ng/VehicleManagement/VerifyPlateNo'

def verify(platenumber):
    data = {'plateNumber': platenumber}
    response = requests.post(url, data=data)
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(response.text)
    table = soup.find('table')
    result = []
    if table:
            columns = soup.find_all('td')  
            for column in columns:
                details = column.find_all('span') 
                for detail in details:
                    result.append(detail.text)

    else:
        result = [f"This number plate {platenumber} is not in the FRSC database."]

    return result

    


def get_plate_number(frame):
    result = ocr_model.predict(frame, conf=0.35)
    
    lst = [(x1,y1,x2,y2,p,c) for x1,y1,x2,y2,p,c in (result[0].boxes.data).numpy()]
    lst.sort()
    plate_lst = [x[5] for x in lst]
    plate = ''
    for c in plate_lst:
        plate += result[0].names[c]

    return plate

#Prediction
def brand_plate_detection(image):
    prediction = brand_plate_model.predict(image) #model.predict(source=..., project="xx", name="xxx")
    plate = "None detected" 
    car_brand = "None detected"
    for x1,y1,x2,y2,p,c in (prediction[0].boxes.data).numpy():
        if c == 8.0:
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # img = cv2.imread(image)
            
            plate_frame = image[y1:y2, x1:x2]
            resized_image = cv2.resize(plate_frame, (450,450))
            # cv2.imwrite('sample.jpg', resized_image)
            result = get_plate_number(resized_image)
            # print(result)
            if len(result) >= 7:
                plate = result
            else:
                plate = "Couldn't read the text on the detected plate"
        elif c in prediction[0].names.keys():
            car_brand = prediction[0].names[c]
            # print(car_brand)
        else:
            car_brand = "Unknown"

        annotated_frame = prediction[0].plot()
    return car_brand, plate, annotated_frame



def detect_car(image):
    results = car_model.predict(image, classes=2)
    x,y,w,h = results[0].boxes.xyxy[0]
    x,y,w,h = int(x),int(y), int(w),int(h)
    plate_frame = image[y:h, x:w]
    return plate_frame

def get_colour_code(img_arr):
    img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3)
    unique_colors, color_counts = np.unique(pixels, axis=0, return_counts=True)
    most_common_color_index = np.argmax(color_counts)
    rgb = unique_colors[most_common_color_index]
    # print(rgb)
    return rgb

def get_color(detected_color):
    color_map = []
    for idx,row in colors.iterrows():
        r = abs(int(detected_color[0]) - row['R'])
        g = abs(int(detected_color[1]) - row['G'])
        b = abs(int(detected_color[2]) - row['B'])

        # Query row values
        color = row['color'], 
        code = row['code']                  #.replace('#', '')

        # Map results
        color_map.append({
                            'color':color, 
                            'code':code,
                            'distance':sum([r,g,b])
                        })
    
    # Get best match (shortest distance)
    best_match = min(color_map, key=lambda x:x['distance'])
    
    # Get color code
    color_code = best_match['code']
    color = colors[colors['code'] == color_code]['color_name'].values[0]
    

    return color

img = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
tmp = tempfile.NamedTemporaryFile(delete=False)
col1, col2 = st.columns(2)
pred_btn = st.button('Predict')
verify_btn = st.button('Verify')
if img:
    tmp.write(img.read())
    try:
        cv_img = cv2.imread(tmp.name)
        col1.image(cv_img, channels='BGR', use_column_width=True)
    
    except:
        st.text("Unable to read image")


if pred_btn:
    if img:
        car_brand, plate, new_img = brand_plate_detection(cv_img)
        colour = get_color(get_colour_code(detect_car(cv_img)))
        
        # plate_num = plate.replace("-","")
        col2.image(new_img, channels='BGR', use_column_width=True)
        st.write("Car Brand:", car_brand)

        st.write("Colour:", colour)

        st.write("Plate Number:", plate)

    else:
        st.text("Please upload an image")

if verify_btn:
    if img:
        plate = brand_plate_detection(cv_img)[1]
        plate_num = plate.strip()
        plate_num = plate_num.replace("-","")
        result = verify(plate_num)
        if len(result) == 2:
            st.write("The details for car with plate number ",plate,"are:")
            st.write("Car Brand:", result[0])
            st.write("Colour:", result[1])
        else:
            st.write(result[0])

    else:
        st.write("Please upload an image")

   