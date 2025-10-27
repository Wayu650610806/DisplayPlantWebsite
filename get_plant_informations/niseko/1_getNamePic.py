import cv2
import os
import numpy as np

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

def get_panel_header_with_tab_box(img, panel_box):
    """
    ครอบตัดส่วนหัวของ Panel, ตรวจจับตำแหน่งกล่องแท็บสีขาว, 
    และส่งคืนภาพที่ครอบตัด (Header ROI) พร้อมพิกัดกล่องแท็บที่ตรวจจับได้
    """
    x_panel, y_panel, w_panel, h_panel = panel_box
    header_height = 50 
    
    if h_panel < header_height: header_height = h_panel
    header_roi_full = img[y_panel : y_panel + header_height, x_panel : x_panel + w_panel]
    
    gray_roi = cv2.cvtColor(header_roi_full, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Fixed Threshold 252
    _, thresh_white = cv2.threshold(gray_roi, 252, 255, cv2.THRESH_BINARY) 

    contours, _ = cv2.findContours(thresh_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_tab_box = None
    max_area = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # เงื่อนไขการกรอง Contour
        if area > 100: # กรอง Noise เล็กๆ
             if w < w_panel * 0.4 and h > header_height * 0.3:
                 if area > max_area:
                     max_area = area
                     best_tab_box = (x, y, w, h)
                     
    # ส่งคืน Header ROI (ภาพที่ครอบตัด 50px แรก) และ Bounding Box ของแท็บ
    return header_roi_full, best_tab_box

# =========================================================================
# 📦 ฟังก์ชันบันทึกภาพ (🔥 ปรับปรุงแล้ว)
# =========================================================================

def save_tab_box_image(header_roi_full, tab_box, base_filename, panel_index, output_dir, padding_px=10):
    """
    ครอบตัดภาพกล่องแท็บที่ตรวจพบ (พร้อมขยายขอบ padding) และบันทึกลงไฟล์
    """
    if tab_box is None:
        return False
        
    x_tab, y_tab, w_tab, h_tab = tab_box
    h_header, w_header = header_roi_full.shape[:2]
    
    # 1. 🔥 คำนวณพิกัดใหม่ (ขยายขอบ) และป้องกันการล้นขอบ
    x1 = max(0, x_tab - padding_px)
    y1 = max(0, y_tab - padding_px)
    x2 = min(w_header, x_tab + w_tab + padding_px)
    y2 = min(h_header, y_tab + h_tab + padding_px)

    # 2. ครอบตัดภาพแท็บจาก Header ROI โดยใช้พิกัดที่ขยายแล้ว
    # [y_start : y_end, x_start : x_end]
    cropped_tab_img = header_roi_full[y1:y2, x1:x2]
    
    if cropped_tab_img.size == 0:
        print(f"⚠️ Error: Cropped image is empty for {base_filename}_P{panel_index}. Check padding/coords.")
        return False

    # 3. สร้างชื่อไฟล์และบันทึก
    filename = f"{os.path.splitext(base_filename)[0]}_P{panel_index}_Tab.png"
    save_path = os.path.join(output_dir, filename)
    
    # การขยายภาพก่อนบันทึกเพื่อ SIFT อาจมีประโยชน์ (เป็นทางเลือก)
    # cropped_tab_img = cv2.resize(cropped_tab_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(save_path, cropped_tab_img)
    return True

# =========================================================================
# 📦 ฟังก์ชันแสดงผลหลัก (🔥 ปรับปรุงแล้ว)
# =========================================================================

def show_images_in_folder(folder, output_folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    if not image_files:
        print("❌ ไม่พบรูปในโฟลเดอร์:", folder)
        return
        
    # สร้างโฟลเดอร์สำหรับบันทึกภาพถ้ายังไม่มี
    os.makedirs(output_folder, exist_ok=True)
    
    # ⚠️ (ปรับค่าได้) กำหนด Padding (พิกเซล) ที่จะขยายรอบกล่องแท็บตอนบันทึก
    TAB_SAVE_PADDING = 80

    idx = 0
    max_display_w, max_display_h = 1600, 900
    
    while True:
        base_filename = image_files[idx]
        path = os.path.join(folder, base_filename)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ โหลดรูปไม่สำเร็จ: {path}")
            idx = (idx + 1) % len(image_files)
            continue

        boxes = split_manual_2_1(img)
        tab_box_info = ""
        
        # 🎯 ตรวจสอบ Panel ที่ 1 และ 2 เพื่อบันทึกภาพแท็บ
        panels_to_process = [1] # สนใจ Panel 2 (Index 1)

        # 1. ครอบตัดและแสดงเฉพาะ Panel 2 Header ROI
        if len(boxes) >= 2:
            
            # ดึง Header ROI และหา Bounding Box ของแท็บ สำหรับ Panel 2
            # 🔥 หมายเหตุ: ผมแก้เป็น boxes[0] นะครับ เพราะคุณสนใจ Panel 1 (Index 0) 
            # ถ้าต้องการ Panel 2 ให้แก้เป็น boxes[1]
            header_roi, tab_box = get_panel_header_with_tab_box(img, boxes[1]) 
            
            # ใช้ภาพ Header ROI เป็นภาพแสดงผลหลัก
            display_img = header_roi.copy() 
            
            # 2. 🔥 บันทึกภาพแท็บ (ส่งค่า Padding เข้าไป)
            # (ผมใช้ panel_index=1 เพื่อให้ชื่อไฟล์เป็น _P1_Tab.png)
            is_saved = save_tab_box_image(header_roi, tab_box, base_filename, 1, 
                                          output_folder, padding_px=TAB_SAVE_PADDING)

            # 3. วาด Bounding Box ของแท็บ (ถ้าตรวจพบ)
            if tab_box:
                x_tab, y_tab, w_tab, h_tab = tab_box
                cv2.rectangle(display_img, (x_tab, y_tab), (x_tab + w_tab, y_tab + h_tab), (0, 0, 255), 2)
                tab_box_info = f"Tab Box W:{w_tab} H:{h_tab} | SAVED: {is_saved}"
            else:
                tab_box_info = "Tab Box: NOT DETECTED"
                
        else:
            display_img = img.copy()
            cv2.putText(display_img, "ERROR: Not enough panels detected!", (10, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # 4. แสดงข้อมูลการตรวจจับ
        cv2.putText(display_img,
                    f"Showing Panel 1 Header ROI | {tab_box_info}", # แก้เป็น Panel 1
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"{base_filename} ({idx+1}/{len(image_files)})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # ย่อภาพให้พอดีหน้าจอ
        h_disp, w_disp = display_img.shape[:2]
        scale = min(max_display_w / w_disp, max_display_h / h_disp, 1.0)
        if scale < 1.0:
            display_img = cv2.resize(display_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        cv2.imshow(f"Panel 1 Header ROI + Tab Detection", display_img) # แก้ชื่อหน้าต่าง
        key = cv2.waitKey(0) & 0xFF

        if key == ord('d'): idx = (idx + 1) % len(image_files)
        elif key == ord('a'): idx = (idx - 1) % len(image_files)
        elif key == ord('q'): break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ⚠️ 1. เปลี่ยน path นี้ให้ชี้ไปยังโฟลเดอร์ที่มีไฟล์รูปภาพของคุณ
    input_folder = r"C:\Project\DisplayPlantWebsite\uploaded_images\tv2\2" 
    
    # ⚠️ 2. กำหนดโฟลเดอร์สำหรับบันทึกภาพแท็บที่ครอบตัดแล้ว
    output_folder = r"C:\Project\DisplayPlantWebsite\output_tab_images"
    
    if not os.path.isdir(input_folder):
        print(f"🚨 Error: โฟลเดอร์ต้นทางไม่พบ: {input_folder}")
    else:
        show_images_in_folder(input_folder, output_folder)