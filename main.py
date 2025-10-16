from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import influxdb_client
from collections import defaultdict
import os
from datetime import datetime

# --- ตั้งค่าการเชื่อมต่อ InfluxDB ---
# INFLUX_URL = "http://localhost:8086"
# INFLUX_TOKEN = "L_YAl3vOfs5L_XebQuSg20uw9H2YkicPi6cXGhopD8d2O-AcwwysqPYiqlc0dgXHp6TwY_etkt5xChYIWrcGxw=="
# INFLUX_ORG = "KinseiPlant"
# INFLUX_BUCKET = "furnace_data"

#------cloud--------
# INFLUX_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
# INFLUX_TOKEN = "oAtmgomIuy4QVTulsgQq8HAwEZmpXBXM5a9rIsiumVbpbwos21uttKuPZWaiKRlIWieU-tkYhAOqNwU8h4SCSg=="
# INFLUX_ORG = "KinseiPlant"
# INFLUX_BUCKET = "furnace_data"

#-------website----------
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = "furnace_data" # Bucket name ไม่ใช่ข้อมูลลับ ใส่ไว้ตรงๆ ได้
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


# --- 3. API สำหรับดึงข้อมูล Plant (โค้ดเดิม) ---
@app.get("/api/plants/overview")
async def get_plants_overview():
    try:
        client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()

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
                if not model:
                    continue

                # ensure dict exists
                if model not in plant_map:
                    plant_map[model] = {
                        "customer": record.values.get("customer"),
                        "province": record.values.get("province") or record.values.get("prefecture") or "",
                        "model": model,
                        "last_updated": None,
                        "sensors": defaultdict(dict)
                    }

                # set last_updated as ISO string (take the latest time we see)
                try:
                    rec_time = record.get_time()
                    if rec_time is not None:
                        # if not set yet or this record is newer -> update ISO string
                        existing = plant_map[model].get("last_updated")
                        if not existing or (rec_time.isoformat() > existing):
                            plant_map[model]["last_updated"] = rec_time.isoformat()
                except Exception:
                    # ignore if cannot parse time
                    pass

                field = record.get_field()
                value = record.get_value()
                sensor_name = record.values.get("sensor_name")

                if field == 'image_url':
                    plant_map[model]['image_url'] = value
                elif sensor_name:
                    # store raw value; frontend code already handles object/value or plain value
                    plant_map[model]['sensors'][sensor_name][field] = value
        
        results = list(plant_map.values())
        return results

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



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