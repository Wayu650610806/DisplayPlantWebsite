import cv2
import os
import numpy as np
import json
from typing import Tuple, Optional, List, Any, Dict

# 🔥 --- Imports สำหรับ OCR (Tesseract เท่านั้น) ---
import re
import pytesseract

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================================================================
# 📦 ฟังก์ชันหลักของการตัดแบ่ง (ไม่เปลี่ยน)
# =========================================================================

def split_manual_2_1(img):
    """
    แบ่งภาพออกเป็น 4 ส่วนเท่าๆ กัน (25% x 4)
    ส่งค่ากลับเป็น list ของ (x, y, w, h)
    """
    h, w = img.shape[:2]
    
    # คำนวณจุดแบ่ง 3 จุด
    split_x_1 = w // 4       # 25%
    split_x_2 = w // 2       # 50%
    split_x_3 = (w * 3) // 4 # 75%
    
    boxes = []
    
    # กล่อง 1: 0% - 25%
    w1 = split_x_1
    boxes.append((0, 0, w1, h))

    # กล่อง 2: 25% - 50%
    x2 = split_x_1
    w2 = split_x_2 - split_x_1 
    boxes.append((x2, 0, w2, h))

    # กล่อง 3: 50% - 75%
    x3 = split_x_2
    w3 = split_x_3 - split_x_2
    boxes.append((x3, 0, w3, h))
    
    # กล่อง 4: 75% - 100% (เก็บส่วนที่เหลือทั้งหมด)
    x4 = split_x_3
    w4 = w - split_x_3
    boxes.append((x4, 0, w4, h))
    
    return boxes


# =========================================================================
# 🔎 ฟังก์ชัน SIFT Matching (ไม่เปลี่ยน)
# =========================================================================

def find_tab_sift(template_img: np.ndarray, target_img: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Any]:
    """
    ใช้ SIFT และ Flann-based Matcher เพื่อค้นหาตำแหน่งของ template_img ใน target_img
    """
    sift = cv2.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(template_img, None)
    kp_target, des_target = sift.detectAndCompute(target_img, None)
    
    if des_template is None or des_target is None or len(des_template) < 2 or len(des_target) < 2:
        return None, 0 

    des_template = des_template.astype(np.float32)
    des_target = des_target.astype(np.float32)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des_template, des_target, k=2)
    except cv2.error:
        return None, 0
        
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 7
    match_count = len(good_matches) # เก็บจำนวน Good Matches
    
    if match_count > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = template_img.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            x_min = int(np.min(dst[:, 0, 0]))
            y_min = int(np.min(dst[:, 0, 1]))
            x_max = int(np.max(dst[:, 0, 0]))
            y_max = int(np.max(dst[:, 0, 1]))
            
            tab_box_sift = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            return tab_box_sift, match_count
        else:
            return None, 0
    else:
        return None, match_count


# =========================================================================
# 🛠️ ฟังก์ชันเสริม (ไม่เปลี่ยน)
# =========================================================================

def check_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    r1 = x1 + w1
    b1 = y1 + h1
    r2 = x2 + w2
    b2 = y2 + h2
    if r1 < x2 or r2 < x1 or b1 < y2 or b2 < y1:
        return False
    return True

def read_image_unicode_path(path: str) -> Optional[np.ndarray]:
    """
    อ่านไฟล์ภาพจากพาธที่มีอักขระ Unicode (เช่น ภาษาญี่ปุ่น) โดยใช้ Buffer
    """
    img_data = np.fromfile(path, np.uint8)
    if img_data.size == 0:
        return None
    return cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# =========================================================================
# 🔥 (NEW) ฟังก์ชันเสริมสำหรับ OCR (Tesseract เท่านั้น)
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
    """โหลดพิกัด Normalized ROIs จากไฟล์ JSON ที่ชื่อเดียวกับ Template"""
    json_filename = f"{os.path.splitext(filename_prefix)[0]}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("rois")
        except Exception as e:
            print(f"❌ Error reading JSON {json_filename}: {e}")
            return None
    return None

# =========================================================================
# ⚙️ ฟังก์ชันประมวลผลหลัก (ไม่เปลี่ยน)
# (ฟังก์ชันนี้เรียก ocr_number() ซึ่งตอนนี้ชี้ไปที่ Tesseract โดยอัตโนมัติ)
# =========================================================================

def show_images_in_folder(folder: str, output_folder: str, status_template_dir: str):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    if not image_files:
        print("❌ ไม่พบรูปในโฟลเดอร์:", folder)
        return
    
    # 🔥 ROI ที่เราจะใช้ SIFT เพื่อหา Template สถานะในภายหลัง (ตรวจสอบจากชื่อ Label)
    STATUS_ROIS = [
        "乾溜ガス化炉A_運転状況", 
        "乾溜ガス化炉B_運転状況", 
        "乾溜ガス化炉C_運転状況"
    ]
    
    # 🔥 1. โหลด TEMPLATES สถานะ ("None", "Cooling", "AUTO")
    STATUS_TEMPLATES = {
        "None": "None.png",
        "Cooling": "Cooling.png",
        "AUTO": "AUTO.png"
    }
    
    loaded_status_templates: Dict[str, np.ndarray] = {}
    
    print("\n--- โหลด Template สถานะ (SIFT Target) ---")
    for name, filename in STATUS_TEMPLATES.items():
        template_path = os.path.join(status_template_dir, filename)
        img = read_image_unicode_path(template_path)
        if img is not None:
            loaded_status_templates[name] = img
            print(f"✅ Loaded: {name} ({filename})")
        else:
            print(f"❌ Failed to load: {name} ({filename})")
    print("-" * 45)
    
    # --- 2. โหลด FIXED TEMPLATE (直富商事.png) และ JSON ก่อนเริ่มลูป ---
    FIXED_TEMPLATE_FILENAME = "loop.png"
    
    fixed_template_path = os.path.join(output_folder, FIXED_TEMPLATE_FILENAME)
    template_img = read_image_unicode_path(fixed_template_path)
    sift_load_status = f"SIFT Template: {'✅' if template_img is not None else '❌'} {FIXED_TEMPLATE_FILENAME} loaded."

    rois_normalized = load_rois_from_json(FIXED_TEMPLATE_FILENAME, output_folder)
    json_load_status = f"JSON ROI: {'✅' + str(len(rois_normalized)) + ' ROIs loaded.' if rois_normalized else '❌ Not Found or Error.'}"
    
    print(sift_load_status)
    print(json_load_status)
    print("=" * 45)
    print("🚀 เริ่มการประมวลผล... (กด 'd' = ถัดไป, 'a' = ก่อนหน้า, 'q' = ออก)")


    idx = 0
    max_display_w, max_display_h = 1600, 900
    
    while True:
        base_filename = image_files[idx]
        path = os.path.join(folder, base_filename)
        img = read_image_unicode_path(path) # Target Image
        
        if img is None:
            print(f"⚠️ โหลดรูปไม่สำเร็จ: {path}")
            idx = (idx + 1) % len(image_files)
            continue
            
        h, w = img.shape[:2] 

        # 🔥 (NEW) พิมพ์ข้อมูลไฟล์ปัจจุบัน
        print(f"\n--- 🏞️ Processing Image {idx+1}/{len(image_files)}: {base_filename} ---")


        # 3. แบ่งภาพเป็น Panel Boxes
        boxes = split_manual_2_1(img) 
        sift_box = None
        
        # 4. SIFT Matching (Fixed Template)
        if template_img is not None:
            sift_box, _ = find_tab_sift(template_img, img)
        
        # 5. ตรวจสอบ Panel ที่ SIFT เจอและจัดการการแสดงผล
        display_img = img.copy() 
        
        sift_status_results: Dict[str, str] = {} 
        ocr_numeric_results: Dict[str, str] = {} 
        
        if sift_box:
            # 🔥 (NEW) พิมพ์ผล SIFT Fixed
            print(f"✅ [SIFT Fixed] DETECTED W:{sift_box[2]} H:{sift_box[3]}")
            
            x_sift_abs, y_sift_abs, w_sift, h_sift = sift_box 
            panel_found_index = -1
            
            # 5.1 ตรวจสอบ Panel ที่ซ้อนทับ
            for i, p_box in enumerate(boxes):
                if check_overlap(sift_box, p_box):
                    panel_found_index = i
                    x_p, y_p, w_p, h_p = p_box 
                    
                    # 5.2 ครอบตัดเฉพาะ Panel ที่พบ (แสดงผล)
                    display_img = img[y_p:y_p+h_p, x_p:x_p+w_p].copy() 
                    print(f"    ➡️  Panel {panel_found_index + 1} (Cropped)")
                    
                    # 5.3 วาด SIFT Box (สีน้ำเงิน) - (คงไว้)
                    x_rel = x_sift_abs - x_p
                    y_rel = y_sift_abs - y_p
                    cv2.rectangle(display_img, (x_rel, y_rel), (x_rel + w_sift, y_rel + h_sift), (255, 0, 0), 2)
                    
                    # 5.4 วาด ROI จาก JSON
                    if rois_normalized:
                        
                        for label, coords_n in rois_normalized.items():
                            
                            x1n, y1n, x2n, y2n = coords_n
                            x1_abs, y1_abs = int(x1n * w), int(y1n * h)
                            x2_abs, y2_abs = int(x2n * w), int(y2n * h)
                            
                            x_roi_rel = x1_abs - x_p
                            y_roi_rel = y1_abs - y_p
                            w_roi = x2_abs - x1_abs
                            h_roi = y2_abs - y1_abs
                            
                            if x_roi_rel < w_p and y_roi_rel < h_p and w_roi > 0 and h_roi > 0:
                                
                                # วาด ROI (สีแดง) - (คงไว้)
                                cv2.rectangle(display_img, (x_roi_rel, y_roi_rel), (x_roi_rel + w_roi, y_roi_rel + h_roi), (0, 0, 255), 2)
                                
                                # 5.5 แยกการทำงาน SIFT (Status) หรือ OCR (Numeric)
                                roi_img = img[y1_abs:y2_abs, x1_abs:x2_abs]

                                if label in STATUS_ROIS:
                                    # --- 5.5a SIFT (สำหรับ Status) ---
                                    best_match = "Unknown"
                                    max_match_count = -1
                                    
                                    for status_name, status_template in loaded_status_templates.items():
                                        _, match_count = find_tab_sift(status_template, roi_img)
                                        
                                        if match_count > max_match_count:
                                            max_match_count = match_count
                                            best_match = status_name
                                            
                                    result_text = f"Status: {best_match} ({max_match_count})"
                                    sift_status_results[label] = result_text
                                    
                                    print(f"    [SIFT Status] {label}: {result_text}")

                                else:
                                    # --- 5.5b OCR (สำหรับตัวเลข) ---
                                    
                                    # 🔥 เรียกใช้ Tesseract Robust
                                    numeric_val, raw_text = ocr_number(roi_img) 
                                    
                                    if numeric_val is not None:
                                        result_text = f"Val: {numeric_val}"
                                        ocr_numeric_results[label] = result_text 
                                    else:
                                        result_text = f"OCR: {raw_text}" 
                                        ocr_numeric_results[label] = result_text
                                    
                                    print(f"    [OCR Numeric] {label}: {result_text}")

                    break # เจอ Panel แรกที่ซ้อนทับแล้ว ออกจากลูป

            if panel_found_index == -1:
                # SIFT เจอ แต่ไม่อยู่ใน 3 กล่องที่เราแบ่งไว้
                cv2.rectangle(display_img, (x_sift_abs, y_sift_abs), (x_sift_abs + w_sift, y_sift_abs + h_sift), (255, 0, 0), 2)
                print("    [SIFT Fixed] Found, but outside main panels.")
                
        else:
            # 🔥 (NEW) พิมพ์ผล SIFT Fixed
            print(f"❌ [SIFT Fixed] NOT FOUND")
            
        
        # 8. ย่อภาพและแสดงผล
        h_disp, w_disp = display_img.shape[:2]
        scale = min(max_display_w / w_disp, max_display_h / h_disp, 1.0)
        if scale < 1.0:
            display_img = cv2.resize(display_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        cv2.imshow(f"Automatic ROI Marking (Panel View)", display_img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('d'): idx = (idx + 1) % len(image_files)
        elif key == ord('a'): idx = (idx - 1) % len(image_files)
        elif key == ord('q'): break

    cv2.destroyAllWindows()
    print("=" * 45)
    print("👋 ปิดโปรแกรม")

# =========================================================================
# 🚀 จุดเริ่มต้นของโปรแกรม
# =========================================================================

if __name__ == "__main__":
    # ⚠️ 1. โฟลเดอร์ที่มีไฟล์รูปภาพต้นฉบับ (Target Images)
    input_folder = r"C:\Project\DisplayPlantWebsite\uploaded_images\tv2\1" 
    
    # ⚠️ 2. โฟลเดอร์ที่มีไฟล์ Template "直富商事.png" และ JSON "直富商事.json" (ใช้เป็น Source ของ JSON ROI เดิม)
    output_folder = r"C:\Project\DisplayPlantWebsite\output_tab_images"
    
    # 🔥 3. โฟลเดอร์ที่มีไฟล์ Template สถานะ "None.png", "Cooling.png", "AUTO.png"
    status_template_dir = r"C:\Project\DisplayPlantWebsite\output_tab_images\status2"
    
    if not os.path.isdir(input_folder):
        print(f"🚨 Error: โฟลเดอร์ต้นทางไม่พบ: {input_folder}")
    elif not os.path.isdir(status_template_dir):
        print(f"🚨 Error: โฟลเดอร์ Template สถานะไม่พบ: {status_template_dir}")
    else:
        show_images_in_folder(input_folder, output_folder, status_template_dir)