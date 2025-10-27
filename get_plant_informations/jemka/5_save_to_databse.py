import cv2
import os
import numpy as np
import json
from typing import Tuple, Optional, List, Any, Dict
import re
import pytesseract
import pprint

# --- Imports สำหรับ InfluxDB ---
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# --- Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- ตั้งค่าการเชื่อมต่อ InfluxDB ---
# (ใส่ค่าจริงของคุณที่นี่)
INFLUX_URL = "https://us-east-1-1.aws.cloud2.influxdata.com" # หรือ URL ของคุณ
INFLUX_TOKEN = "oAtmgomIuy4QVTulsgQq8HAwEZmpXBXM5a9rIsiumVbpbwos21uttKuPZWaiKRlIWieU-tkYhAOqNwU8h4SCSg==" # ใส่ Token ของคุณ
INFLUX_ORG = "KinseiPlant" # ใส่ Org ของคุณ
INFLUX_BUCKET = "plant_data" # ใส่ Bucket ของคุณ

# =========================================================================
# 📦 ฟังก์ชันหลักของการตัดแบ่ง (ไม่เปลี่ยน)
# =========================================================================
def split_manual_2_1(img):
    """
    แบ่งภาพออกเป็น 3 ส่วนเท่าๆ กันในแนวตั้ง (ความกว้างเท่ากัน)
    ส่งค่ากลับเป็น list ของ (x, y, w, h)
    """
    h, w = img.shape[:2]
    
    # คำนวณจุดแบ่ง 2 จุด
    split_x_1 = w // 3       # ประมาณ 33.3%
    split_x_2 = (w * 2) // 3 # ประมาณ 66.7%
    
    boxes = []
    
    # กล่อง 1: 0% - 33.3%
    x1 = 0
    w1 = split_x_1
    boxes.append((x1, 0, w1, h))

    # กล่อง 2: 33.3% - 66.7%
    x2 = split_x_1
    w2 = split_x_2 - split_x_1
    boxes.append((x2, 0, w2, h))

    # กล่อง 3: 66.7% - 100% (เก็บส่วนที่เหลือทั้งหมด)
    # ใช้ w - split_x_2 เพื่อเก็บเศษพิกเซลที่เหลือจากการหารไม่ลงตัว
    x3 = split_x_2
    w3 = w - split_x_2 
    boxes.append((x3, 0, w3, h))
    
    return boxes


# =========================================================================
# 🔎 ฟังก์ชัน SIFT Matching (ไม่เปลี่ยน)
# =========================================================================
def find_tab_sift(template_img: np.ndarray, target_img: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Any]:
    sift = cv2.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(template_img, None)
    kp_target, des_target = sift.detectAndCompute(target_img, None)
    if des_template is None or des_target is None or len(des_template) < 2 or len(des_target) < 2: return None, 0
    des_template = des_template.astype(np.float32)
    des_target = des_target.astype(np.float32)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try: matches = flann.knnMatch(des_template, des_target, k=2)
    except cv2.error: return None, 0
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: good_matches.append(m)
    MIN_MATCH_COUNT = 7
    match_count = len(good_matches)
    if match_count > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = template_img.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            x_min, y_min = int(np.min(dst[:, 0, 0])), int(np.min(dst[:, 0, 1]))
            x_max, y_max = int(np.max(dst[:, 0, 0])), int(np.max(dst[:, 0, 1]))
            tab_box_sift = (x_min, y_min, x_max - x_min, y_max - y_min)
            return tab_box_sift, match_count
        else: return None, 0
    else: return None, match_count

# =========================================================================
# 🛠️ ฟังก์ชันเสริม (ไม่เปลี่ยน)
# =========================================================================
def check_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    x1, y1, w1, h1 = box1; x2, y2, w2, h2 = box2
    r1, b1 = x1 + w1, y1 + h1; r2, b2 = x2 + w2, y2 + h2
    if r1 < x2 or r2 < x1 or b1 < y2 or b2 < y1: return False
    return True

def read_image_unicode_path(path: str) -> Optional[np.ndarray]:
    img_data = np.fromfile(path, np.uint8)
    if img_data.size == 0: return None
    return cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# =========================================================================
# 🔥 ฟังก์ชัน OCR (Robust V6 - Smart Tournament)
# =========================================================================
number_re = re.compile(r"-?\d+(?:[.,]\d+)?")

def ocr_number(bgr: np.ndarray) -> Tuple[Optional[float], str]:
    """
    (Robust Version 6) "Anti 5->6 Tournament"
    - ตัดการ DILATE (ทำให้หนา) ทิ้งทั้งหมด ซึ่งเป็นสาเหตุหลักของ 5 -> 6
    - เพิ่มการ MORPH_OPEN (เปิด) เพื่อช่วยลบ Noise โดยไม่ไปอุดช่องว่างของเลข 5
    - ยังคงใช้ Confidence-based (image_to_data) ในการเลือกผู้ชนะ
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if max(g.shape) < 60:
        g = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    def parse_and_get_confidence(img: np.ndarray, config: str) -> Tuple[Optional[float], int, str]:
        """
        รัน OCR และคืนค่า (ตัวเลขที่แปลงแล้ว, confidence, raw_text)
        """
        try:
            # ใช้ image_to_data เพื่อเอา confidence
            d = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            
            # หา "word" ที่มี confidence สูงสุด
            if 'text' in d and len(d['text']) > 0:
                best_conf = -1
                best_text = ""
                for i in range(len(d['text'])):
                    # เลือกเฉพาะ word ที่มีตัวอักษรและ confidence > 0
                    if d['text'][i].strip() and int(d['conf'][i]) > best_conf:
                        best_conf = int(d['conf'][i])
                        best_text = d['text'][i].strip()

                if not best_text:
                    return None, -1, ""

                # ทำความสะอาด text ที่ได้
                t = best_text.replace("O", "0").replace("o", "0").replace("S", "5").strip()
                m = number_re.search(t)
                
                if not m:
                    return None, best_conf, t
                
                s = m.group().replace(",", ".")
                try:
                    return float(s), best_conf, t
                except ValueError:
                    return None, best_conf, t
            return None, -1, ""
        except Exception:
            return None, -1, ""

    # 1. กำหนด Configs
    # (เพิ่ม 'S' เข้า whitelist เพราะบางครั้ง 5 ถูกอ่านเป็น S)
    cfg_7 = f"--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.-S" 
    cfg_13 = f"--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789.-S"

    # 2. Kernel (สำหรับ Opening)
    kernel_open = np.ones((2, 2), np.uint8)

    # 3. สร้างภาพสำหรับ "การแข่งขัน" (ลบ Dilate, เพิ่ม Open)
    
    # แบบ Inverted (ตัวอักษรขาว พื้นดำ)
    otsu_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    adapt_inv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # แบบ Normal (ตัวอักษรดำ พื้นขาว - Tesseract มักจะชอบ)
    otsu_norm = cv2.bitwise_not(otsu_inv) 
    adapt_norm = cv2.bitwise_not(adapt_inv)
    
    # 🔥 (NEW) แบบ Opening (ลบ Noise แต่ไม่ปิดช่องว่าง)
    # เราทำ Opening กับภาพแบบ Inverted (ที่ตัวอักษรเป็นสีขาว)
    otsu_inv_opened = cv2.morphologyEx(otsu_inv, cv2.MORPH_OPEN, kernel_open)
    adapt_inv_opened = cv2.morphologyEx(adapt_inv, cv2.MORPH_OPEN, kernel_open)


    # รายการภาพที่จะนำไปทดสอบ (Image, Description)
    image_candidates = [
        (otsu_inv, "Otsu Inverted (Thin)"),
        (otsu_norm, "Otsu Normal (Thin)"),
        (adapt_inv, "Adaptive Inverted (Thin)"),
        (adapt_norm, "Adaptive Normal (Thin)"),
        (otsu_inv_opened, "Otsu Inverted (Opened)"), # 🔥 ตัวใหม่
        (adapt_inv_opened, "Adaptive Inverted (Opened)") # 🔥 ตัวใหม่
    ]

    # 4. 🔥 รัน "การแข่งขันแบบ Confidence-based"
    results = []
    # print("--- OCR Tournament (V6) ---") # (เอา comment ออก ถ้าต้องการ Debug)
    
    for img, desc in image_candidates:
        # ลองกับ psm 7 และ 13 สำหรับแต่ละภาพ
        res7 = parse_and_get_confidence(img, cfg_7)
        res13 = parse_and_get_confidence(img, cfg_13)
        
        # (Debug output)
        # if res7[2]: print(f"  - {desc} (psm 7):  '{res7[2]}' (Conf: {res7[1]}) -> {res7[0]}")
        # if res13[2]: print(f"  - {desc} (psm 13): '{res13[2]}' (Conf: {res13[1]}) -> {res13[0]}")
        
        results.append(res7)
        results.append(res13)

    # 5. เลือกผู้ชนะ
    # 5.1 กรองเอาเฉพาะผลลัพธ์ที่แปลงเป็นตัวเลขได้
    valid_results = [r for r in results if r[0] is not None]

    if not valid_results:
        # ถ้าไม่มีอันไหนแปลงเป็นเลขได้เลย, คืนค่า raw text ที่ confidence สูงสุด
        results.sort(key=lambda x: x[1], reverse=True) # เรียงตาม conf
        all_raw = results[0][2] if results else ""
        return (None, all_raw)
    
    # 5.2 ถ้ามี, เลือกอันที่ confidence สูงที่สุด
    valid_results.sort(key=lambda x: x[1], reverse=True) # เรียงตาม conf
    best_result = valid_results[0]
    
    # print(f"🏆 OCR Winner: {best_result[0]} (from raw '{best_result[2]}' with conf {best_result[1]})")
    return (best_result[0], best_result[2])

# =========================================================================
# 💾 ฟังก์ชันโหลด JSON (ไม่เปลี่ยน)
# =========================================================================
def load_rois_from_json(filename_prefix: str, output_dir: str) -> Optional[dict]:
    json_filename = f"{os.path.splitext(filename_prefix)[0]}.json"
    json_path = os.path.join(output_dir, json_filename)
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
            return data.get("rois")
        except Exception as e: print(f"❌ Error reading JSON {json_filename}: {e}"); return None
    return None

# =========================================================================
# 🔥 ฟังก์ชันเขียนข้อมูลลง InfluxDB (ไม่เปลี่ยน)
# =========================================================================
def write_data(write_api, tags, fields, timestamp):
    if not any(value is not None for value in fields.values()):
        print(f"    ⚠️ Skipping write for {tags.get('sensor_name')} - all fields are None.")
        return False
    if timestamp is None:
        print(f"    ⚠️ Skipping write for {tags.get('sensor_name')} - timestamp is None.")
        return False
    try:
        if not isinstance(timestamp, datetime):
             raise TypeError(f"Timestamp must be a datetime object, got {type(timestamp)}")
        point = influxdb_client.Point("plant_information").time(timestamp, write_precision="s")
        for key, value in tags.items():
            if value is not None: point.tag(key, str(value))
        for key, value in fields.items():
            if value is not None:
                try: point.field(key, float(value))
                except (ValueError, TypeError): point.field(key, str(value))
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        return True
    except Exception as e:
        print(f"    ❌ Error writing data to InfluxDB for {tags.get('sensor_name')}: {e}")
        return False

# =========================================================================
# ⚙️ ฟังก์ชันประมวลผลหลัก (เวอร์ชันเขียนลง InfluxDB)
# =========================================================================
def show_images_in_folder(folder: str, output_folder: str, status_template_dir: str):
    image_files_raw = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files_raw: print("❌ ไม่พบรูปในโฟลเดอร์:", folder); return

    def get_number_key(filename):
        num_str = re.search(r'(\d+)', filename); return int(num_str.group(1)) if num_str else 0
    image_files = sorted(image_files_raw, key=get_number_key)
    print(f"✅ พบ {len(image_files)} รูปภาพ (เรียงลำดับตามตัวเลขแล้ว)")

    STATUS_ROIS = ["乾溜ガス化炉A_運転状況", "乾溜ガス化炉B_運転状況", "乾溜ガス化炉C_運転状況"]
    STATUS_TEMPLATES = {"None": "None.png", "Cooling": "Cooling.png", "AUTO": "AUTO.png"}
    loaded_status_templates: Dict[str, np.ndarray] = {}
    print("\n--- โหลด Template สถานะ (SIFT Target) ---")
    for name, filename in STATUS_TEMPLATES.items():
        template_path = os.path.join(status_template_dir, filename); img_tpl = read_image_unicode_path(template_path)
        if img_tpl is not None: loaded_status_templates[name] = img_tpl; print(f"✅ Loaded: {name} ({filename})")
        else: print(f"❌ Failed to load: {name} ({filename})")
    print("-" * 45)

    FIXED_TEMPLATE_FILENAME = "ジェムカ.png"
    fixed_template_path = os.path.join(output_folder, FIXED_TEMPLATE_FILENAME)
    template_img = read_image_unicode_path(fixed_template_path)
    print(f"SIFT Template: {'✅' if template_img is not None else '❌'} {FIXED_TEMPLATE_FILENAME} loaded.")
    rois_normalized = load_rois_from_json(FIXED_TEMPLATE_FILENAME, output_folder)
    print(f"JSON ROI: {'✅' + str(len(rois_normalized)) + ' ROIs loaded.' if rois_normalized else '❌ Not Found or Error.'}")

    # --- โหลด Metadata ---
    timestamp_json_path = os.path.join(output_folder, "timestamps_jemka.json"); all_timestamps = {}
    if os.path.exists(timestamp_json_path):
        try:
            with open(timestamp_json_path, 'r', encoding='utf-8') as f: all_timestamps = json.load(f)
            print(f"✅ Loaded {len(all_timestamps)} timestamps from timestamps.json")
        except Exception as e: print(f"❌ Error loading timestamps.json: {e}")
    else: print(f"⚠️ Warning: timestamps.json not found at {timestamp_json_path}")

    plant_data_json_path = os.path.join(output_folder, "plant_data.json"); all_plant_data = {}; customer_info = {}
    if os.path.exists(plant_data_json_path):
        try:
            with open(plant_data_json_path, 'r', encoding='utf-8') as f: all_plant_data = json.load(f)
            print(f"✅ Loaded {len(all_plant_data)} plant records from plant_data.json")
            customer_key = os.path.splitext(FIXED_TEMPLATE_FILENAME)[0]; customer_info = all_plant_data.get(customer_key)
            if customer_info: print(f"    ➡️  Using data for customer: {customer_info.get('customer')}")
            else: print(f"❌ Error: Cannot find customer key '{customer_key}' in plant_data.json"); customer_info = {}
        except Exception as e: print(f"❌ Error loading plant_data.json: {e}")
    else: print(f"⚠️ Warning: plant_data.json not found at {plant_data_json_path}")

    # --- สร้าง Client และ Write API ของ InfluxDB ---
    client = None
    write_api = None
    try:
        print("\n--- 🔌 Connecting to InfluxDB ---")
        client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        client.health()
        print("    ✅ Connection Successful!")
    except Exception as e:
        print(f"    ❌ FAILED to connect to InfluxDB: {e}")
        print("    ⚠️ Will proceed without writing data.")
        client = None
        write_api = None

    print("=" * 45)
    print("🚀 เริ่มการประมวลผล... (กด 'd' = ถัดไป, 'a' = ก่อนหน้า, 'q' = ออก)")

    idx = 0
    max_display_w, max_display_h = 1600, 900

    while True:
        base_filename = image_files[idx]
        path = os.path.join(folder, base_filename)
        img = read_image_unicode_path(path)
        if img is None: print(f"⚠️ โหลดรูปไม่สำเร็จ: {path}"); idx = (idx + 1) % len(image_files); continue

        h, w = img.shape[:2]
        print(f"\n--- 🏞️ Processing Image {idx+1}/{len(image_files)}: {base_filename} ---")

        # --- ดึงและแปลง Timestamp ---
        image_timestamp_str = all_timestamps.get(base_filename)
        image_timestamp_dt = None
        if image_timestamp_str:
            try:
                image_timestamp_dt = datetime.fromisoformat(image_timestamp_str)
                image_timestamp_dt = image_timestamp_dt.astimezone(timezone.utc)
                print(f"    🕒 Timestamp found: {image_timestamp_str} -> UTC: {image_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except ValueError:
                print(f"    ⚠️ Invalid timestamp format in JSON: '{image_timestamp_str}'. Skipping timestamp.")
                image_timestamp_dt = None
        else:
            print(f"    ⚠️ No timestamp found for '{base_filename}'. Skipping timestamp.")
            image_timestamp_dt = None

        # --- SIFT Matching ---
        boxes = split_manual_2_1(img)
        sift_box = None
        if template_img is not None: sift_box, _ = find_tab_sift(template_img, img)
        display_img = img.copy()

        if sift_box:
            print(f"✅ [SIFT Fixed] DETECTED W:{sift_box[2]} H:{sift_box[3]}")
            x_sift_abs, y_sift_abs, w_sift, h_sift = sift_box
            panel_found_index = -1

            for i, p_box in enumerate(boxes):
                if check_overlap(sift_box, p_box):
                    panel_found_index = i
                    x_p, y_p, w_p, h_p = p_box
                    display_img = img[y_p:y_p+h_p, x_p:x_p+w_p].copy()
                    print(f"    ➡️  Panel {panel_found_index + 1} (Cropped)")
                    x_rel, y_rel = x_sift_abs - x_p, y_sift_abs - y_p
                    cv2.rectangle(display_img, (x_rel, y_rel), (x_rel + w_sift, y_rel + h_sift), (255, 0, 0), 2)

                    # --- Pass 1 + Pass 2 ---
                    temp_storage: Dict[str, float] = {}
                    grouped_data = defaultdict(lambda: {"tags": {}, "fields": {}})

                    if rois_normalized:
                        for label, coords_n in rois_normalized.items():
                            x1n, y1n, x2n, y2n = coords_n; x1_abs, y1_abs = int(x1n * w), int(y1n * h); x2_abs, y2_abs = int(x2n * w), int(y2n * h)
                            x_roi_rel, y_roi_rel = x1_abs - x_p, y1_abs - y_p; w_roi, h_roi = x2_abs - x1_abs, y2_abs - y1_abs

                            if x_roi_rel < w_p and y_roi_rel < h_p and w_roi > 0 and h_roi > 0:
                                cv2.rectangle(display_img, (x_roi_rel, y_roi_rel), (x_roi_rel + w_roi, y_roi_rel + h_roi), (0, 0, 255), 2)
                                roi_img = img[y1_abs:y2_abs, x1_abs:x2_abs]

                                parts = label.split('_')
                                sensor_name, field_name, unit = (None, None, None)
                                if label in STATUS_ROIS and len(parts) == 2: sensor_name, field_name = parts
                                elif len(parts) == 3: sensor_name, field_name, unit = parts
                                else: sensor_name = label
                                if not sensor_name: continue

                                group_key = (customer_info.get('customer'), customer_info.get('model'), sensor_name)
                                if not grouped_data[group_key]['tags']:
                                     grouped_data[group_key]['tags'] = {
                                         "customer": customer_info.get('customer'),
                                         "province": customer_info.get('province'),
                                         "model": customer_info.get('model'),
                                         "sensor_name": sensor_name
                                     }

                                if label in STATUS_ROIS:
                                    best_match = "Unknown"; max_match_count = -1
                                    for status_name, status_template in loaded_status_templates.items():
                                        _, match_count = find_tab_sift(status_template, roi_img)
                                        if match_count > max_match_count: max_match_count = match_count; best_match = status_name
                                    print(f"    [SIFT Status] {label}: Status: {best_match} ({max_match_count})")
                                    grouped_data[group_key]['fields'][f'_{field_name}_raw'] = best_match
                                else:
                                    numeric_val, raw_text = ocr_number(roi_img) # ใช้ V6
                                    if numeric_val is not None: print(f"    [OCR Numeric] {label}: Val: {numeric_val}")
                                    else: print(f"    [OCR Numeric] {label}: OCR: {raw_text}")
                                    field_key = f"{field_name}_{unit}" if unit else field_name
                                    grouped_data[group_key]['fields'][field_key] = numeric_val
                                    if field_name == '温度': temp_storage[sensor_name] = numeric_val

                    # --- Pass 2.5: แปลง Status ---
                    for group in grouped_data.values():
                        sensor_name = group['tags'].get('sensor_name')
                        fields = group['fields']
                        raw_status = fields.pop(f'_運転状況_raw', None)
                        if raw_status:
                            temp_value = temp_storage.get(sensor_name, 999.0)
                            if temp_value is None: temp_value = 999.0
                            final_status = None
                            if raw_status == 'AUTO': final_status = 'AUTO'
                            elif raw_status == 'Cooling': final_status = '冷却'
                            elif raw_status == 'None':
                                if temp_value < 40: final_status = '投入・灰出'
                                else: final_status = '冷却'
                            fields['運転状況'] = final_status

                    # --- เขียนข้อมูลลง InfluxDB ---
                    if write_api:
                        print("\n    --- ☁️ Writing data to InfluxDB ---")
                        write_count = 0
                        for group in grouped_data.values():
                            if write_data(write_api, tags=group['tags'], fields=group['fields'], timestamp=image_timestamp_dt):
                                write_count += 1
                        print(f"    ☁️ Wrote {write_count} points to InfluxDB.")
                    else:
                        print("\n    ⚠️ Skipping InfluxDB write (connection failed earlier).")
                        print("    --- 📊 GROUPED DATA (for debugging) ---")
                        pprint.pprint(dict(grouped_data))
                    break # ออกจากลูป Panel

            if panel_found_index == -1: print("    [SIFT Fixed] Found, but outside main panels.")
        else: print(f"❌ [SIFT Fixed] NOT FOUND")

        # --- แสดงผลภาพ ---
        h_disp, w_disp = display_img.shape[:2]
        scale = min(max_display_w / w_disp, max_display_h / h_disp, 1.0)
        if scale < 1.0: display_img = cv2.resize(display_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(f"Automatic ROI Marking (Panel View)", display_img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'): idx = (idx + 1) % len(image_files)
        elif key == ord('a'): idx = (idx - 1) % len(image_files)
        elif key == ord('q'): break

    # --- ปิดการเชื่อมต่อ ---
    if client:
        print("\n--- 🔌 Closing InfluxDB connection ---")
        client.close()
    cv2.destroyAllWindows()
    print("=" * 45)
    print("👋 ปิดโปรแกรม")

# =========================================================================
# 🚀 จุดเริ่มต้นของโปรแกรม
# =========================================================================
if __name__ == "__main__":
    input_folder = r"C:\Project\DisplayPlantWebsite\uploaded_images\tv1\2"
    output_folder = r"C:\Project\DisplayPlantWebsite\output_tab_images" # แก้ Path ตามที่คุณใช้
    status_template_dir = r"C:\Project\DisplayPlantWebsite\output_tab_images\status_jemka" # แก้ Path ตามที่คุณใช้

    if not os.path.isdir(input_folder): print(f"🚨 Error: โฟลเดอร์ต้นทางไม่พบ: {input_folder}")
    elif not os.path.isdir(status_template_dir): print(f"🚨 Error: โฟลเดอร์ Template สถานะไม่พบ: {status_template_dir}")
    else: show_images_in_folder(input_folder, output_folder, status_template_dir)