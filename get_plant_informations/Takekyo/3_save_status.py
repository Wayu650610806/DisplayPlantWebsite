import cv2
import os
import numpy as np
import json
from typing import Tuple, Optional, List, Any

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
        return None, None 

    des_template = des_template.astype(np.float32)
    des_target = des_target.astype(np.float32)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des_template, des_target, k=2)
    except cv2.error:
        return None, None
        
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 5

    if len(good_matches) > MIN_MATCH_COUNT:
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
            
            return tab_box_sift, (kp_template, kp_target, good_matches, mask, M, dst)
        else:
            return None, None
    else:
        return None, None

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
# 💾 ฟังก์ชันโหลด JSON และบันทึกรูป ROI
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

def save_cropped_roi(img: np.ndarray, coords_n: List[float], roi_label: str, output_dir: str):
    """
    ครอบตัด ROI จากภาพต้นฉบับ (img) และบันทึกโดยใช้ชื่อ label
    """
    try:
        h, w = img.shape[:2]
        x1n, y1n, x2n, y2n = coords_n
        
        # แปลงพิกัด Normalized เป็น Absolute
        x1_abs, y1_abs = int(x1n * w), int(y1n * h)
        x2_abs, y2_abs = int(x2n * w), int(y2n * h)
        
        # ครอบตัดภาพ
        cropped_roi = img[y1_abs:y2_abs, x1_abs:x2_abs]
        
        if cropped_roi.size > 0:
            # สร้างชื่อไฟล์โดยใช้ Label
            filename = f"{roi_label}.png"
            save_path = os.path.join(output_dir, filename)
            
            # ใช้วิธี numpy/imencode เพื่อรองรับชื่อไฟล์ Unicode (ภาษาญี่ปุ่น)
            is_success, im_buf_arr = cv2.imencode(".png", cropped_roi)
            if is_success:
                # ใช้ tofile เพื่อให้รองรับ Unicode path
                im_buf_arr.tofile(save_path) 
                return f"Saved: {filename}"
            else:
                return f"Error: Failed to encode image {filename}."
        else:
            return "Error: Cropped ROI is empty (check coordinates)."
            
    except Exception as e:
        return f"Error during cropping/saving ROI '{roi_label}': {e}"


# =========================================================================
# ⚙️ ฟังก์ชันประมวลผลหลัก
# =========================================================================

def show_images_in_folder(folder: str, output_folder: str):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    if not image_files:
        print("❌ ไม่พบรูปในโฟลเดอร์:", folder)
        return
    
    # 🔥 ROI ที่ต้องการบันทึกเป็นรูปภาพสำหรับ SIFT Template ในอนาคต (ตรวจสอบจากชื่อ Label)
    ROIS_TO_SAVE = [
        "乾溜ガス化炉A_運転状況", 
        "乾溜ガス化炉B_運転状況"
    ]
    
    # --- 1. โหลด FIXED TEMPLATE และ JSON ก่อนเริ่มลูป ---
    FIXED_TEMPLATE_FILENAME = "武京商会.png"
    
    fixed_template_path = os.path.join(output_folder, FIXED_TEMPLATE_FILENAME)
    template_img = read_image_unicode_path(fixed_template_path)
    sift_load_status = f"SIFT Template: {'✅' if template_img is not None else '❌'} {FIXED_TEMPLATE_FILENAME} loaded."

    rois_normalized = load_rois_from_json(FIXED_TEMPLATE_FILENAME, output_folder)
    json_load_status = f"JSON ROI: {'✅' + str(len(rois_normalized)) + ' ROIs loaded.' if rois_normalized else '❌ Not Found or Error.'}"
    
    # พิมพ์ชื่อไฟล์ที่จะบันทึกหาก SIFT เจอ
    if rois_normalized:
        print("\nROIs ที่จะถูกบันทึกเป็นไฟล์ภาพ (Template สำหรับ SIFT ในอนาคต):")
        for roi_name in ROIS_TO_SAVE:
             if roi_name in rois_normalized:
                 print(f" - {roi_name}.png")
             else:
                 print(f" - ⚠️ {roi_name}.png (ไม่พบใน JSON)")
        print("-" * 30)


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
            
        h, w = img.shape[:2] # ได้ขนาดภาพต้นฉบับ

        # 2. แบ่งภาพเป็น Panel Boxes
        boxes = split_manual_2_1(img) 
        sift_box = None
        sift_info = "SIFT: Match Not Attempted"
        
        # 3. SIFT Matching
        if template_img is not None:
            sift_box, _ = find_tab_sift(template_img, img)
        
        # 4. ตรวจสอบ Panel ที่ SIFT เจอและจัดการการแสดงผล
        display_img = img.copy() 
        display_text = "Match Failed."
        
        if sift_box:
            sift_info = f"SIFT: DETECTED W:{sift_box[2]} H:{sift_box[3]}"
            x_sift_abs, y_sift_abs, w_sift, h_sift = sift_box 
            panel_found_index = -1
            
            # 4.1 ตรวจสอบ Panel ที่ซ้อนทับ
            for i, p_box in enumerate(boxes):
                if check_overlap(sift_box, p_box):
                    panel_found_index = i
                    x_p, y_p, w_p, h_p = p_box 
                    
                    # 🔥 4.2 ครอบตัดเฉพาะ Panel ที่พบ (แสดงผล)
                    display_img = img[y_p:y_p+h_p, x_p:x_p+w_p].copy() 
                    display_text = f"Panel {panel_found_index + 1} (Cropped)"
                    
                    # 4.3 วาด SIFT Box (สีน้ำเงิน)
                    x_rel = x_sift_abs - x_p
                    y_rel = y_sift_abs - y_p
                    cv2.rectangle(display_img, (x_rel, y_rel), (x_rel + w_sift, y_rel + h_sift), (255, 0, 0), 2)
                    
                    # 🔥 4.4 วาด ROI จาก JSON และบันทึกรูปภาพที่ต้องการ
                    if rois_normalized:
                        
                        save_status_info = [] 
                        
                        for label, coords_n in rois_normalized.items():
                            
                            # แปลงพิกัด Normalized เป็นพิกัดสัมบูรณ์
                            x1n, y1n, x2n, y2n = coords_n
                            x1_abs, y1_abs = int(x1n * w), int(y1n * h)
                            x2_abs, y2_abs = int(x2n * w), int(y2n * h)
                            
                            # แปลงพิกัดสัมบูรณ์เป็นพิกัดสัมพัทธ์ (ใน Panel ที่ถูกตัด)
                            x_roi_rel = x1_abs - x_p
                            y_roi_rel = y1_abs - y_p
                            w_roi = x2_abs - x1_abs
                            h_roi = y2_abs - y1_abs
                            
                            # ตรวจสอบว่า ROI นี้อยู่ใน Panel ที่ถูกตัดหรือไม่
                            if x_roi_rel < w_p and y_roi_rel < h_p:
                                
                                # วาด ROI (สีแดง)
                                cv2.rectangle(display_img, (x_roi_rel, y_roi_rel), (x_roi_rel + w_roi, y_roi_rel + h_roi), (0, 0, 255), 2)
                                
                                # แสดง Label บนภาพ
                                cv2.putText(display_img, label, (x_roi_rel, y_roi_rel - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                            
                                # 🔥 บันทึกรูปภาพ ROI ที่ต้องการ (ตรวจสอบจากชื่อ Label)
                                if label in ROIS_TO_SAVE:
                                    save_result = save_cropped_roi(img, coords_n, label, output_folder)
                                    # เพื่อให้ข้อความสถานะไม่ยาวเกินไป ให้แสดงแค่ Saved/Error
                                    save_status_info.append(f"Save '{label}': {save_result.split(':')[0]}") 
                        
                        if save_status_info:
                            # แสดงสถานะการบันทึกใน console
                            print(f"\n--- Saving ROIs for {base_filename} ---")
                            print("\n".join(save_status_info))
                            print("-" * 30)
                            
                    break # เจอ Panel แรกที่ซ้อนทับแล้ว ออกจากลูป

            if panel_found_index == -1:
                # SIFT เจอ แต่ไม่อยู่ใน 3 กล่องที่เราแบ่งไว้
                cv2.rectangle(display_img, (x_sift_abs, y_sift_abs), (x_sift_abs + w_sift, y_sift_abs + h_sift), (255, 0, 0), 2)
                display_text = "SIFT Found, but outside main panels."
                
        else:
            sift_info = "SIFT: NOT FOUND"
            display_text = "Match Failed."
            

        # 5. แสดงข้อมูลการตรวจจับ
        # แสดงสถานะการโหลด Template และ JSON
        cv2.putText(display_img, sift_load_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, json_load_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # แสดงผลลัพธ์ SIFT และสถานะ
        cv2.putText(display_img, 
                    f"{base_filename} ({idx+1}/{len(image_files)}) | {sift_info} | Showing: {display_text}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 6. ย่อภาพและแสดงผล
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

# =========================================================================
# 🚀 จุดเริ่มต้นของโปรแกรม
# =========================================================================

if __name__ == "__main__":
    # ⚠️ 1. โฟลเดอร์ที่มีไฟล์รูปภาพต้นฉบับ (Target Images)
    input_folder = r"C:\Project\DisplayPlantWebsite\uploaded_images\tv2\3" 
    
    # ⚠️ 2. โฟลเดอร์ที่มีไฟล์ Template "鈴木工業.png" และ JSON "鈴木工業.json"
    output_folder = r"C:\Project\DisplayPlantWebsite\output_tab_images"
    
    if not os.path.isdir(input_folder):
        print(f"🚨 Error: โฟลเดอร์ต้นทางไม่พบ: {input_folder}")
    else:
        show_images_in_folder(input_folder, output_folder)