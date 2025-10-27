from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import influxdb_client
from collections import defaultdict
from pathlib import Path
import os, time, secrets, shutil, json
from influxdb_client.client.exceptions import InfluxDBError

# --- ตั้งค่าการเชื่อมต่อ InfluxDB ---
# INFLUX_URL = "http://localhost:8086"
# INFLUX_TOKEN = "L_YAl3vOfs5L_XebQuSg20uw9H2YkicPi6cXGhopD8d2O-AcwwysqPYiqlc0dgXHp6TwY_etkt5xChYIWrcGxw=="
# INFLUX_ORG = "KinseiPlant"
# INFLUX_BUCKET = "furnace_data"

#------cloud--------
INFLUX_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUX_TOKEN = "Xttrq8yiXo5GrzZ5p6J2AxzXKYDEniqO9_3fzD_3Zt9fAbalTW1Cbtjt-mjfb9TZuSa-mK8_Iovea-dyIegQ-A=="
INFLUX_ORG = "KinseiPlant" # <-- ใช้ค่านี้
INFLUX_BUCKET = "plant_data"

#-------website----------
# INFLUX_URL = os.getenv("INFLUX_URL")
# INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
# INFLUX_ORG = os.getenv("INFLUX_ORG")
# INFLUX_BUCKET = "furnace_data" # Bucket name ไม่ใช่ข้อมูลลับ ใส่ไว้ตรงๆ ได้

BASE_DIR = Path(__file__).parent # Path ปัจจุบันของ main.py
UPLOAD_DIR = BASE_DIR / "uploaded_images" # โฟลเดอร์สำหรับเก็บรูปภาพ
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # สร้าง folder ถ้ายังไม่มี
LATEST_IMAGE_JSON = BASE_DIR / "latest_image.json" # ไฟล์สำหรับบันทึก URL รูปภาพล่าสุด


# --- สร้าง FastAPI App ---
app = FastAPI()


# --- ตั้งค่า CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount โฟลเดอร์ uploaded_images (สำหรับภาพที่อัปโหลด)
app.mount("/uploaded_images", StaticFiles(directory=str(UPLOAD_DIR)), name="uploaded_images")

# 2. Mount โฟลเดอร์ static (สำหรับ CSS, JS, Fonts, logo.png ฯลฯ)
static_dir = "static"
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    # โค้ดนี้จะรันเมื่อโฟลเดอร์ static ไม่มีอยู่
    print(f"คำเตือน: ไม่พบโฟลเดอร์ '{static_dir}' สำหรับไฟล์ static")

    
# --- 1. ทำให้ Backend รู้จักโฟลเดอร์ "static" ---
static_dir = "static"
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"คำเตือน: ไม่พบโฟลเดอร์ '{static_dir}' สำหรับไฟล์ static")


# --- 2. สร้าง API สำหรับหน้า Home (/) ---
@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/api/plants/overview")
async def get_plants_overview():
    try:
        # สมมติว่า INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET และ defaultdict ถูก import แล้ว
        client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        # *** NOTE: ควรใช้ Flux Query ที่ละเอียดกว่านี้ เพื่อดึงข้อมูล plant ทุกรายการออกมา ***
        # แต่เพื่อการสาธิตการใช้คีย์ผสม จะใช้ Query เดิมก่อน
        query = f'''
            from(bucket: "{INFLUX_BUCKET}")
            |> range(start: -30d)
            |> filter(fn: (r) => r["_measurement"] == "plant_information")
            |> last() 
        '''
        tables = query_api.query(query, org=INFLUX_ORG)
        client.close()

        plant_map = {}
        for table in tables:
            for record in table.records:
                # แก้ไข: ถ้า model เป็น None ให้ใช้ "" แทน เพื่อให้สร้างคีย์ได้
                model = record.values.get("model") or "" 
                customer = record.values.get("customer") or "" # ป้องกัน customer เป็น None ด้วย
                province = record.values.get("province") or record.values.get("prefecture") or "" 
                
                # --- ✨ ส่วนที่แก้ไข: สร้างคีย์ผสม ✨ ---
                # ใช้ tuple ของค่าที่สำคัญเป็นคีย์ (model, customer, province)
                unique_key = (model, customer, province)
                
                # ลบบรรทัด 'if not model: continue' ออก เพื่อให้ Plant ที่ไม่มี model ถูกประมวลผล
                
                # --- ใช้คีย์ผสมในการตรวจสอบและเพิ่มข้อมูล ---
                if unique_key not in plant_map:
                    plant_map[unique_key] = {
                        "customer": customer,
                        "province": province,
                        "model": model, # model จะเป็น "" ถ้าไม่มีค่า
                        "last_updated": None, 
                        "sensors": defaultdict(dict)
                    }
                
                # ใช้คีย์ผสมในการอ้างอิงข้อมูล
                rec_time = record.get_time()
                if rec_time:
                    existing = plant_map[unique_key]["last_updated"]
                    if not existing or rec_time.isoformat() > existing:
                        plant_map[unique_key]["last_updated"] = rec_time.isoformat()
                        
                field = record.get_field()
                value = record.get_value()
                sensor_name = record.values.get("sensor_name")
                
                if field == "image_url":
                    plant_map[unique_key]["image_url"] = value
                elif sensor_name:
                    plant_map[unique_key]["sensors"][sensor_name][field] = value

        # --- ส่วนสุดท้าย: คืนค่าเป็น List ของ values (ข้อมูล Plant) ---
        return list(plant_map.values()) 
    except Exception as e:
        # ตรวจสอบว่าได้ import JSONResponse มาแล้ว (เช่น from fastapi.responses import JSONResponse)
        return JSONResponse({"error": str(e)}, status_code=500)
    
@app.get("/api/plants/overview")
async def get_plants_overview():
    try:
        # สมมติว่า INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET และ defaultdict ถูก import แล้ว
        
        query_api = client.query_api()
        
        # *** NOTE: ควรใช้ Flux Query ที่ละเอียดกว่านี้ เพื่อดึงข้อมูล plant ทุกรายการออกมา ***
        # แต่เพื่อการสาธิตการใช้คีย์ผสม จะใช้ Query เดิมก่อน
        query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: -30d)
              |> filter(fn: (r) => r["_measurement"] == "plant_information")
              |> last() 
        '''
        tables = query_api.query(query, org=INFLUX_ORG)
        client.close()

        plant_map = {}
        for table in tables:
            for record in table.records:
                model = record.values.get("model")
                customer = record.values.get("customer") # ดึง customer มาใช้ในคีย์
                province = record.values.get("province") or record.values.get("prefecture") or "" # ดึง province มาใช้ในคีย์
                # url = record.values.get("image_url")
                # --- ✨ ส่วนที่แก้ไข: สร้างคีย์ผสม ✨ ---
                # ใช้ tuple ของค่าที่สำคัญเป็นคีย์ (model, customer, province)
                unique_key = (model, customer, province)
                
                if not model: continue
                
                # --- ใช้คีย์ผสมแทน model ในการตรวจสอบและเพิ่มข้อมูล ---
                if unique_key not in plant_map:
                    plant_map[unique_key] = {
                        "customer": customer,
                        "province": province,
                        "model": model, 
                        "last_updated": None, 
                        "sensors": defaultdict(dict)
                    }
                
                # ใช้คีย์ผสมในการอ้างอิงข้อมูล
                rec_time = record.get_time()
                if rec_time:
                    existing = plant_map[unique_key]["last_updated"]
                    if not existing or rec_time.isoformat() > existing:
                        plant_map[unique_key]["last_updated"] = rec_time.isoformat()
                        
                field = record.get_field()
                value = record.get_value()
                sensor_name = record.values.get("sensor_name")
                
                if field == "image_url":
                    plant_map[unique_key]["image_url"] = value
                elif sensor_name:
                    plant_map[unique_key]["sensors"][sensor_name][field] = value

        # --- ส่วนสุดท้าย: คืนค่าเป็น List ของ values (ข้อมูล Plant) ---
        return list(plant_map.values()) 
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- 4. ✨ API ใหม่: สำหรับดึงข้อมูลย้อนหลังเพื่อทำกราฟ (ปรับปรุงใหม่) ---
@app.get("/api/plant/{model_name}/history")
async def get_plant_history(model_name: str, range_hours: int = 6):
    """
    API สำหรับดึงข้อมูลย้อนหลังของ Plant ที่ระบุตามช่วงเวลา (ชั่วโมง)
    นำกลับเป็น dict grouped by sensor_name:
      { "sensorA": [ {time, field, value, unit}, ... ], ... }
    ถ้า point ใน Influx มี tag/column ชื่อ 'unit' จะถูกใส่ลงในแต่ละ record
    """
    try:
        client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()

        query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: -{range_hours}h)
              |> filter(fn: (r) => r["_measurement"] == "plant_information")
              |> filter(fn: (r) => r["model"] == "{model_name}")
              |> filter(fn: (r) => r["_field"] == "温度_℃" or r["_field"] == "開度_%")
              |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
              |> yield(name: "results")
        '''

        tables = query_api.query(query, org=INFLUX_ORG)
        client.close()

        history_data = defaultdict(list)
        for table in tables:
            for record in table.records:
                # sensor_name ควรอยู่ใน tag values ตามตัวอย่างของคุณ
                sensor_name = record.values.get("sensor_name")
                if not sensor_name:
                    continue

                # พยายามอ่าน unit ถ้ามี (อาจเป็น tag หรือ column)
                unit = None
                # common places to find unit: custom tag 'unit', 'unit' in values, หรือ 'field_name' มี unit แยก
                if "unit" in record.values:
                    unit = record.values.get("unit")
                else:
                    # บางกรณี unit อาจอยู่ใน 'field_name' หรือใน _field เป็น "温度_℃"
                    # ถ้า _field มีสัญลักษณ์หน่วยให้ใช้นั้นเป็น fallback
                    _field = record.get_field() or ""
                    if "℃" in _field or "温度" in _field:
                        unit = "°C"
                    elif "%" in _field or "開度" in _field:
                        unit = "%"

                # เก็บ time เป็น ISO format, value เป็น numeric
                time_iso = None
                try:
                    t = record.get_time()
                    time_iso = t.isoformat() if t is not None else None
                except Exception:
                    time_iso = None

                history_data[sensor_name].append({
                    "time": time_iso,
                    "field": record.get_field(),
                    "value": record.get_value(),
                    "unit": unit
                })

        # (option) เรียงแต่ละ sensor ตามเวลา ascending เพื่อความแน่นอน
        for sensor, recs in history_data.items():
            recs.sort(key=lambda r: r.get("time") or "")

        return history_data

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/detail.html")
async def read_detail():
    # ตรวจสอบว่ามีไฟล์ detail.html อยู่จริงหรือไม่ ก่อนที่จะส่งกลับไป
    if os.path.exists("detail.html"):
        return FileResponse("detail.html")
    return JSONResponse(status_code=404, content={"error": "detail.html not found"})

# --- 5. ✨ API ใหม่: สำหรับรับและบันทึกรูปภาพ ---
@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_ext = Path(file.filename).suffix
        name = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4) + file_ext
        outpath = UPLOAD_DIR / name
        with open(outpath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_url = f"/uploaded_images/{name}"
        with open(LATEST_IMAGE_JSON, "w", encoding="utf-8") as f:
            json.dump({"latest_url": image_url, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        return {"message": "success", "file_url": image_url}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- 6. API สำหรับหน้าแสดงผลภาพล่าสุด ---
@app.get("/latest_image", response_class=FileResponse)
async def serve_latest_image_page():
    """
    ส่งไฟล์ HTML ใหม่ (latest_image_page.html) กลับไป
    """
    # ตรวจสอบว่ามีไฟล์ HTML อยู่จริงหรือไม่ ก่อนที่จะส่งกลับไป
    html_path = BASE_DIR / "latest_image_page.html"
    if html_path.exists():
        return FileResponse(html_path)
    
    # ถ้าไม่มีไฟล์ HTML ให้แจ้งเตือน
    return JSONResponse(status_code=404, content={"error": "latest_image_page.html not found. Please create it."})


# --- 7. API สำหรับดึง URL รูปภาพล่าสุด (ใช้โดย Frontend JS) ---
@app.get("/api/latest_image_url")
async def get_latest_image_url():
    if LATEST_IMAGE_JSON.exists():
        try:
            return json.load(open(LATEST_IMAGE_JSON, "r", encoding="utf-8"))
        except:
            return JSONResponse({"latest_url": None, "error": "Cannot read JSON file"}, status_code=500)
    return {"latest_url": None, "timestamp": None}
