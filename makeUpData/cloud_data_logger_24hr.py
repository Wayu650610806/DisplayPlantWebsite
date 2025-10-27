# data_logger.py (24-Hour Historical Data Generator)

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import random
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# --- ตั้งค่าการเชื่อมต่อ (เหมือนเดิม) ---
INFLUX_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUX_TOKEN = "Xttrq8yiXo5GrzZ5p6J2AxzXKYDEniqO9_3fzD_3Zt9fAbalTW1Cbtjt-mjfb9TZuSa-mK8_Iovea-dyIegQ-A=="
INFLUX_ORG = "KinseiPlant"
INFLUX_BUCKET = "plant_data"

# --- ฟังก์ชัน write_data (เหมือนเดิม) ---
def write_data(write_api, tags, fields, timestamp):
    if not any(value is not None for value in fields.values()): return False
    try:
        point = influxdb_client.Point("plant_information").time(timestamp)
        for key, value in tags.items():
            if value is not None: point.tag(key, value)
        for key, value in fields.items():
            if value is not None:
                try: point.field(key, float(value))
                except (ValueError, TypeError): point.field(key, value)
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        return True
    except Exception as e:
        print(f"❌ Error writing data: {e}")
        return False

# --- ✨ 1. ฟังก์ชันใหม่สำหรับสร้างข้อมูลดิบแบบสุ่ม ---
def generate_random_plant_data():
    """สร้างข้อมูลดิบ 1 ชุด (สำหรับ 1 timestamp) แบบสุ่มทั้งหมด"""
    statuses = ['AUTO', '投入・灰出', '冷却']
    
    # ข้อมูลดิบทั้งหมด โดยแทนที่ค่าด้วย random
    raw_data = [
        # --- 武京商会 ---
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜ガス化炉A', 'field_name': '温度', 'unit': '℃', 'value': round(random.uniform(250, 400), 2), 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜ガス化炉A', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': random.choice(statuses)},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜ガス化炉B', 'field_name': '温度', 'unit': '℃', 'value': round(random.uniform(10, 200), 2), 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜ガス化炉B', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': random.choice(statuses)},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜空気弁A', 'field_name': '開度', 'unit': '%', 'value': random.randint(0, 100), 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '乾溜空気弁B', 'field_name': '開度', 'unit': '%', 'value': random.randint(0, 100), 'status': None},
        {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '燃焼炉', 'field_name': '温度', 'unit': '℃', 'value': round(random.uniform(800, 1200), 2), 'status': None},
        # {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '排ガス濃度', 'field_name': 'CO濃度', 'unit': 'ppm', 'value': random.randint(0, 50), 'status': None},
        # {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型', 'sensor_name': '排ガス濃度', 'field_name': 'O2濃度', 'unit': '%', 'value': round(random.uniform(5, 15), 2), 'status': None},
        # --- 光陽建設 ---
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜ガス化炉A', 'field_name': '温度', 'unit': '℃', 'value': round(random.uniform(280, 420), 2), 'status': None},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜ガス化炉A', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': random.choice(statuses)},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜ガス化炉B', 'field_name': '温度', 'unit': '℃', 'value': round(random.uniform(10, 220), 2), 'status': None},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜ガス化炉B', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': random.choice(statuses)},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜ガス化炉C', 'field_name': '温度', 'unit': '℃', 'value': round(random.uniform(450, 600), 2), 'status': None},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜ガス化炉C', 'field_name': '運転状況', 'unit': None, 'value': None, 'status': random.choice(statuses)},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜空気弁A', 'field_name': '開度', 'unit': '%', 'value': random.randint(0, 100), 'status': None},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜空気弁B', 'field_name': '開度', 'unit': '%', 'value': random.randint(0, 100), 'status': None},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '乾溜空気弁C', 'field_name': '開度', 'unit': '%', 'value': random.randint(0, 100), 'status': None},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '燃焼炉', 'field_name': '温度', 'unit': '℃', 'value': round(random.uniform(900, 1300), 2), 'status': None},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '排ガス濃度', 'field_name': 'CO濃度', 'unit': 'ppm', 'value': random.randint(0, 60), 'status': None},
        {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型', 'sensor_name': '排ガス濃度', 'field_name': 'O2濃度', 'unit': '%', 'value': round(random.uniform(4, 12), 2), 'status': None},
    ]
    return raw_data

# --- ✨ 2. ส่วนหลักในการทำงาน (ปรับปรุงใหม่ทั้งหมด) ---
def main():
    client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    print("เริ่มต้นโปรแกรมสร้างข้อมูลย้อนหลัง 24 ชั่วโมง...")
    
    # ล้างข้อมูลเก่าทิ้งก่อน
    # clear_bucket_data() # หากต้องการลบข้อมูลเก่าก่อนรัน ให้เปิด comment บรรทัดนี้

    # กำหนดช่วงเวลา
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=24)
    current_time = now

    # วนลูปสร้างข้อมูลย้อนหลัง
    while current_time > start_time:
        print(f"กำลังสร้างข้อมูลสำหรับเวลา: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. สร้างข้อมูลดิบแบบสุ่ม
        raw_data_points = generate_random_plant_data()

        # 2. จัดกลุ่มข้อมูล (เหมือนเดิม)
        grouped_data = defaultdict(lambda: {"tags": {}, "fields": {}})
        for dp in raw_data_points:
            key = (dp['customer'], dp['model'], dp['sensor_name'])
            grouped_data[key]['tags'] = {"customer": dp['customer'], "province": dp['province'], "model": dp['model'], "sensor_name": dp['sensor_name']}
            field_name = dp['field_name']
            if dp['value'] is not None:
                field_key = f"{field_name}_{dp['unit']}" if dp['unit'] else field_name
                grouped_data[key]['fields'][field_key] = dp['value']
            if dp['status'] is not None:
                grouped_data[key]['fields'][field_name] = dp['status']
        
        # 3. ส่งข้อมูลที่จัดกลุ่มแล้วเข้า InfluxDB
        for group in grouped_data.values():
            write_data(write_api, tags=group['tags'], fields=group['fields'], timestamp=current_time)

        # ลดเวลาลง 5 นาทีสำหรับรอบถัดไป
        current_time -= timedelta(minutes=30)

    # client.close()
    print("\n✨ สร้างข้อมูลย้อนหลัง 24 ชั่วโมงสำเร็จ!")

if __name__ == "__main__":
    main()