import cv2
import os
import numpy as np
import json
from typing import Tuple, Optional, List, Any

# =========================================================================
# 📦 ฟังก์ชันหลักของการตัดแบ่ง (ไม่เปลี่ยน)
# =========================================================================

def split_manual_2_1(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    แบ่งภาพออกเป็น 3 ส่วน: กล่อง 1 (25%), กล่อง 2 (25%), กล่อง 3 (50%)
    """
    h, w = img.shape[:2]
    split_x_1 = w // 2
    split_x_2 = split_x_1 // 2
    boxes = []
    
    # กล่อง 1: ซ้ายสุด (Index 0) - (25%)
    w1 = split_x_2
    boxes.append((0, 0, w1, h))

    # กล่อง 2: กลาง (Index 1) - (25%)
    x2 = split_x_2
    w2 = split_x_1 - split_x_2 
    boxes.append((x2, 0, w2, h))

    # กล่อง 3: ขวาใหญ่สุด (Index 2) - (50%)
    x3 = split_x_1
    w3 = w - split_x_1 
    boxes.append((x3, 0, w3, h))
    
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

    MIN_MATCH_COUNT = 10

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
# 🔥 ฟังก์ชันใหม่: การวาด ROI แบบโต้ตอบและบันทึก JSON
# =========================================================================

def interactive_roi_marking(cropped_img: np.ndarray, panel_box_abs: Tuple[int, int, int, int], sift_box_rel: Tuple[int, int, int, int], img_size: Tuple[int, int], filename_prefix: str, output_dir: str) -> bool:
    """
    เปิดหน้าต่างเพื่อให้ผู้ใช้ลาก ROI บนภาพที่ถูกครอบตัด
    """
    h_panel, w_panel = cropped_img.shape[:2]
    img_w_abs, img_h_abs = img_size
    x_panel_abs, y_panel_abs, _, _ = panel_box_abs

    rois = {}  # {label: [x1n,y1n,x2n,y2n]}
    
    print("\n========================================================")
    print("🎯 MODE: INTERACTIVE ROI MARKING")
    print("วิธีใช้: ลาก selectROI -> กด Enter ยืนยัน -> พิมพ์ label ใน console -> Enter อีกครั้ง")
    print("กด q ที่หน้าต่างภาพ เพื่อจบการเลือก ROI และบันทึก JSON")
    print("========================================================")
    
    temp_img = cropped_img.copy()

    while True:
        # วาด SIFT Box (สีน้ำเงิน) เป็นการอ้างอิงบนภาพที่จะลาก ROI
        x_sift, y_sift, w_sift, h_sift = sift_box_rel
        cv2.rectangle(temp_img, (x_sift, y_sift), (x_sift + w_sift, y_sift + h_sift), (255, 0, 0), 2)
        
        cv2.putText(temp_img, "Draw ROI (Blue Box is SIFT Match)", (10, h_panel - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 1. เลือก ROI (ใช้ภาพที่ถูกครอบตัดแล้ว)
        r = cv2.selectROI("Interactive ROI Marker (Panel View)", temp_img, False, False)
        cv2.destroyAllWindows()
        x_rel, y_rel, w_rel, h_rel = map(int, r)
        
        # ตรวจสอบการออกจากการเลือก (เช่น กด Q)
        if w_rel == 0 or h_rel == 0:
            break

        # 2. ใส่ label
        label = input("Label สำหรับ ROI นี้ (เช่น Temp_A, Fan_Hz): ").strip()
        if not label:
            label = f"roi_{len(rois)+1}"
        
        # 3. คำนวณพิกัด Normalized (ต้องเป็นพิกัดสัมบูรณ์เทียบกับภาพต้นฉบับ W, H)
        
        # แปลงพิกัดสัมพัทธ์ (ใน Panel) เป็นพิกัดสัมบูรณ์ (ในภาพต้นฉบับ)
        x_abs = x_panel_abs + x_rel
        y_abs = y_panel_abs + y_rel
        w_abs = w_rel
        h_abs = h_rel

        # Normalized Coordinates (เทียบกับภาพต้นฉบับทั้งหมด)
        x1n, y1n = x_abs / img_w_abs, y_abs / img_h_abs
        x2n, y2n = (x_abs + w_abs) / img_w_abs, (y_abs + h_abs) / img_h_abs
        
        # 4. บันทึก ROI
        rois[label] = [x1n, y1n, x2n, y2n]
        print(f"บันทึก {label} -> Normalized: [{x1n:.4f}, {y1n:.4f}, {x2n:.4f}, {y2n:.4f}]")
        
        # วาด ROI ล่าสุดลงบนภาพชั่วคราว (สีแดง)
        cv2.rectangle(temp_img, (x_rel, y_rel), (x_rel + w_rel, y_rel + h_rel), (0, 0, 255), 2)
        
    # --- 5. บันทึก JSON เมื่อออกจากลูป ---
    
    if rois:
        json_data = {
            "image_size": [img_w_abs, img_h_abs],
            "rois": rois
        }
        
        # สร้างชื่อไฟล์ JSON (直富商事.json)
        json_filename = f"{os.path.splitext(filename_prefix)[0]}.json"
        save_path = os.path.join(output_dir, json_filename)
        
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"✔ FINAL JSON Saved: {json_filename} ({len(rois)} ROIs)")
            return True
        except Exception as e:
            print(f"❌ JSON Save Error: {e}")
            return False
    else:
        print("ℹ️ No ROI was marked. JSON not saved.")
        return False


# =========================================================================
# ⚙️ ฟังก์ชันแสดงผลและประมวลผลหลัก (ใช้ FIXED TEMPLATE และตัดภาพ)
# =========================================================================

def show_images_in_folder(folder: str, output_folder: str):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    if not image_files:
        print("❌ ไม่พบรูปในโฟลเดอร์:", folder)
        return
    
    # --- 1. โหลด FIXED TEMPLATE ---
    FIXED_TEMPLATE_FILENAME = "直富商事.png"
    fixed_template_path = os.path.join(output_folder, FIXED_TEMPLATE_FILENAME)
    template_img = None
    
    if os.path.exists(fixed_template_path):
        template_img = read_image_unicode_path(fixed_template_path)
        if template_img is None:
            sift_load_status = f"SIFT Template: ⚠️ Error reading {FIXED_TEMPLATE_FILENAME}."
        else:
            sift_load_status = f"SIFT Template: ✅ {FIXED_TEMPLATE_FILENAME} loaded."
    else:
        sift_load_status = f"SIFT Template: ❌ File {FIXED_TEMPLATE_FILENAME} NOT FOUND."

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
        boxes = split_manual_2_1(img) # [Box 0, Box 1, Box 2]
        sift_box = None
        sift_info = "SIFT: Match Not Attempted"
        panel_found_index = -1
        display_img = img.copy() 
        display_text = "Original Image"
        json_saved = False
        
        # 3. SIFT Matching
        if template_img is not None:
            sift_box, _ = find_tab_sift(template_img, img)
        
        # 4. ตรวจสอบ Panel ที่ SIFT เจอและจัดการการแสดงผล
        if sift_box:
            sift_info = f"SIFT: DETECTED W:{sift_box[2]} H:{sift_box[3]}"
            x_sift_abs, y_sift_abs, w_sift, h_sift = sift_box # พิกัดสัมบูรณ์

            # 4.1 ตรวจสอบ Panel ที่ซ้อนทับ
            for i, p_box in enumerate(boxes):
                if check_overlap(sift_box, p_box):
                    panel_found_index = i
                    x_p, y_p, w_p, h_p = p_box # พิกัด Panel Box
                    
                    # 🔥 4.2 ครอบตัดเฉพาะ Panel ที่พบ 
                    cropped_img = img[y_p:y_p+h_p, x_p:x_p+w_p].copy() 
                    
                    # 4.3 คำนวณพิกัด SIFT Box สัมพัทธ์ในภาพที่ถูกครอบตัด
                    x_rel = x_sift_abs - x_p
                    y_rel = y_sift_abs - y_p
                    sift_box_rel = (x_rel, y_rel, w_sift, h_sift)

                    # 4.4 เรียกใช้โหมดวาด ROI แบบโต้ตอบ!
                    print(f"\n*** SIFT found tab in Panel {panel_found_index + 1} ***")
                    json_saved = interactive_roi_marking(
                        cropped_img, 
                        p_box, 
                        sift_box_rel, 
                        (w, h), 
                        FIXED_TEMPLATE_FILENAME, 
                        output_folder
                    )
                    
                    # 4.5 เตรียมภาพแสดงผล (ใช้ภาพเต็มพร้อมกรอบ Panel เพื่อแสดงว่า ROI ถูกกำหนดแล้ว)
                    display_img = img.copy()
                    cv2.rectangle(display_img, (x_p, y_p), (x_p + w_p, y_p + h_p), (0, 0, 255), 4) # Panel (สีแดง)
                    display_text = f"Panel {panel_found_index + 1} Marked. Check JSON."
                        
                    break # เจอ Panel แรกก็พอ
            
            if panel_found_index == -1:
                cv2.rectangle(display_img, (x_sift_abs, y_sift_abs), (x_sift_abs + w_sift, y_sift_abs + h_sift), (255, 0, 0), 2)
                display_text = "SIFT Found, but outside main panels. No JSON."
                
        else:
            sift_info = "SIFT: NOT FOUND"
            display_text = "Match Failed."
            

        # 5. แสดงข้อมูลการตรวจจับ
        cv2.putText(display_img, sift_load_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, 
                    f"{base_filename} ({idx+1}/{len(image_files)}) | {sift_info}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_img, 
                    f"Status: {display_text}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 6. ย่อภาพและแสดงผล
        h_disp, w_disp = display_img.shape[:2]
        scale = min(max_display_w / w_disp, max_display_h / h_disp, 1.0)
        if scale < 1.0:
            display_img = cv2.resize(display_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # ⚠️ เปลี่ยนชื่อหน้าต่างให้ตรงกับการแสดงภาพเต็ม
        cv2.imshow(f"SIFT Result & ROI Status (Full Image)", display_img)
        
        # ⚠️ ใช้ waitKey(0) เพื่อรอการกดปุ่มเปลี่ยนรูปถัดไป
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
    input_folder = r"C:\Project\DisplayPlantWebsite\uploaded_images" 
    
    # ⚠️ 2. โฟลเดอร์ที่มีไฟล์ Template "直富商事.png" และจะเซฟ JSON ไว้
    output_folder = r"C:\Project\DisplayPlantWebsite\output_tab_images"
    
    if not os.path.isdir(input_folder):
        print(f"🚨 Error: โฟลเดอร์ต้นทางไม่พบ: {input_folder}")
    else:
        show_images_in_folder(input_folder, output_folder)