 # data_logger.py

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import random
import time
from datetime import datetime, timezone
from collections import defaultdict

# --- ตั้งค่าการเชื่อมต่อ InfluxDB (แก้ไขให้เป็นข้อมูลของคุณ) ---
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "L_YAl3vOfs5L_XebQuSg20uw9H2YkicPi6cXGhopD8d2O-AcwwysqPYiqlc0dgXHp6TwY_etkt5xChYIWrcGxw=="  # << ❗ ใส่ Token ที่คัดลอกมาที่นี่
INFLUX_ORG = "KinseiPlant"         # << ❗ ใส่ชื่อ Organization ของคุณ
INFLUX_BUCKET = "furnace_data"   # << ❗ ใส่ชื่อ Bucket ของคุณ

# ==============================================================================
# ## ฟังก์ชันสำหรับเพิ่มข้อมูล (คุณสามารถปรับแก้เนื้อหาในนี้ได้) ##
# ==============================================================================
# แก้ไขฟังก์ชันนี้
def write_data(write_api, tags, fields, timestamp):
    """
    ฟังก์ชันสำหรับเขียนข้อมูลที่ถูกจัดกลุ่มแล้ว พร้อมระบุเวลา (Timestamp) ที่ต้องการ
    """
    if not any(value is not None for value in fields.values()):
        return False
    try:
        point = influxdb_client.Point("plant_information")

        # --- ✨ การเปลี่ยนแปลงที่ 1: บังคับใช้เวลาที่เราส่งเข้ามา ---
        point.time(timestamp)

        # ที่เหลือเหมือนเดิม
        for key, value in tags.items():
            if value is not None:
                point.tag(key, value)
        for key, value in fields.items():
            if value is not None:
                try:
                    point.field(key, float(value))
                except (ValueError, TypeError):
                    point.field(key, value)

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        print(f"✅ เพิ่มข้อมูลเวลา {timestamp.strftime('%Y-%m-%d %H:%M:%S')} สำเร็จ!")
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเพิ่มข้อมูล: {e}")
        return False
    

def clear_bucket_data():
    """ฟังก์ชันสำหรับลบข้อมูลทั้งหมดใน Bucket ที่กำหนด"""
    try:
        client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        delete_api = client.delete_api()

        start_time = "1970-01-01T00:00:00Z"
        stop_time = datetime.now(timezone.utc).isoformat()
        
        # สร้างเงื่อนไข (predicate) เพื่อระบุว่าจะลบข้อมูลจาก measurement ไหน
        predicate = '_measurement="plant_information"' # <--- ✨ บรรทัดที่เพิ่มเข้ามา

        print(f"กำลังลบข้อมูลทั้งหมดใน bucket '{INFLUX_BUCKET}'...")
        # เพิ่ม predicate เข้าไปในคำสั่ง delete
        delete_api.delete(start=start_time, stop=stop_time, predicate=predicate, bucket=INFLUX_BUCKET, org=INFLUX_ORG)

        client.close()
        print("✨ ลบข้อมูลสำเร็จ!")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
# --- ส่วนหลักในการทำงาน ---
# --- ส่วนหลักในการทำงาน ---
def main():
    client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    print("เริ่มต้นโปรแกรม...")
    # clear_bucket_data()
    model_specs = [
        {
            'tags': {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型'},
            # --- ✨ แก้ไขตรงนี้ ✨ ---
            'fields': {'image_url': '/static/plant/GB-30W-6000特型.png'}
        },
        {
            'tags': {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型'},
            # --- ✨ แก้ไขตรงนี้ ✨ ---
            'fields': {'image_url': '/static/plant/GB-125T-32000特型.png'}
        }
    ]
    # clear_bucket_data()
    # 1. ข้อมูลดิบทั้งหมดที่คุณเตรียมไว้ (เหมือนเดิม)
    raw_data_points = [
        # --- 武京商会 ---
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜ガス化炉A', 'field_name': '温度', 'unit': '℃', 'value': 99, 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜ガス化炉A', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': 'AUTO'},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜ガス化炉B', 'field_name': '温度', 'unit': '℃', 'value': 1000, 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜ガス化炉B', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': '投入・灰出'},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜空気弁A', 'field_name': '開度', 'unit': '%', 'value': 12, 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜空気弁B', 'field_name': '開度', 'unit': '%', 'value': 103, 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '燃焼炉', 'field_name': '温度', 'unit': '℃', 'value': 4, 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '排ガス濃度', 'field_name': 'CO濃度', 'unit': 'ppm', 'value': 13, 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '排ガス濃度', 'field_name': 'O2濃度', 'unit': '%', 'value': 21, 'status': None},
        # --- 光陽建設 ---
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜ガス化炉A', 'field_name': '温度', 'unit': '℃', 'value': 777, 'status': None},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜ガス化炉A', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': 'AUTO'},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜ガス化炉B', 'field_name': '温度', 'unit': '℃', 'value': 9, 'status': None},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜ガス化炉B', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': '投入・灰出'},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜ガス化炉C', 'field_name': '温度', 'unit': '℃', 'value': 3090, 'status': None},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜ガス化炉C', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': '冷却'},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜空気弁A', 'field_name': '開度', 'unit': '%', 'value': 9, 'status': None},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜空気弁B', 'field_name': '開度', 'unit': '%', 'value': 50, 'status': None},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '乾溜空気弁C', 'field_name': '開度', 'unit': '%', 'value': 100, 'status': None},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '燃焼炉', 'field_name': '温度', 'unit': '℃', 'value': 1000, 'status': None},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '排ガス濃度', 'field_name': 'CO濃度', 'unit': 'ppm', 'value': 1, 'status': None},
        {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型', 'sensor_name': '排ガス濃度', 'field_name': 'O2濃度', 'unit': '%', 'value': 1, 'status': None},
    ]

    # 2. จัดกลุ่มข้อมูลตาม "Key" ที่ซ้ำกัน (customer, model, sensor_name)
    # เพื่อรวมข้อมูลให้อยู่ใน record เดียวกัน
    grouped_data = defaultdict(lambda: {"tags": {}, "fields": {}})
    for dp in raw_data_points:
        # สร้าง Key ที่ไม่ซ้ำกันสำหรับแต่ละ Sensor
        key = (dp['customer'], dp['model'], dp['sensor_name'])
        
        # เก็บ Tags (จะถูกเขียนทับด้วยค่าเดิมซึ่งไม่เป็นไร เพราะมันคือค่าเดียวกัน)
        grouped_data[key]['tags'] = {
            "customer": dp['customer'], "province": dp['province'], 
            "model": dp['model'], "sensor_name": dp['sensor_name']
        }
        
        # เพิ่ม Fields ใหม่เข้าไปในกลุ่ม
        field_name = dp['field_name']
        if dp['value'] is not None:
            # ทำให้ชื่อ Field มีหน่วยอยู่ในตัวเลย เพื่อง่ายต่อการดูใน Grafana
            # เช่น "温度_℃", "開度_%"
            field_key = f"{field_name}_{dp['unit']}" if dp['unit'] else field_name
            grouped_data[key]['fields'][field_key] = dp['value']
        if dp['status'] is not None:
            # สำหรับ status เราใช้ชื่อ field_name ตรงๆ
            grouped_data[key]['fields'][field_name] = dp['status']

   
    # --- ✨ 4. ส่งข้อมูลทั้งหมดเข้า InfluxDB ---
    master_timestamp = datetime.now(timezone.utc)
    print(f"\n--- กำลังส่งข้อมูลด้วยเวลา: {master_timestamp.strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # ส่งข้อมูลจำเพาะของแต่ละ Model (ที่มี Image URL)
    for spec in model_specs:
        write_data(write_api, tags=spec['tags'], fields=spec['fields'], timestamp=master_timestamp)

    # ส่งข้อมูล Sensor ที่จัดกลุ่มแล้ว
    for group in grouped_data.values(): # <--- ✅ แก้ไขเป็นชื่อนี้
        write_data(write_api, tags=group['tags'], fields=group['fields'], timestamp=master_timestamp)


    client.close()
    print("\nจบการทำงาน")

if __name__ == "__main__":
    main()