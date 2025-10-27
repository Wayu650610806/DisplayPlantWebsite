# data_logger_fixed.py

import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

# --- ใช้ env vars แทนการ hardcode token/url/org/bucket ---
INFLUX_URL = os.getenv("INFLUX_URL", "https://us-east-1-1.aws.cloud2.influxdata.com")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "oAtmgomIuy4QVTulsgQq8HAwEZmpXBXM5a9rIsiumVbpbwos21uttKuPZWaiKRlIWieU-tkYhAOqNwU8h4SCSg==")  # แนะนำตั้งค่าใน env แทนใส่ตรงนี้
INFLUX_ORG = os.getenv("INFLUX_ORG", "KinseiPlant")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "plant_data")


def write_data(write_api, tags, fields, timestamp):
    """
    เขียน Point โดยใช้ timestamp ที่ส่งเข้าไป
    คืนค่า True/False ว่าเขียนสำเร็จหรือไม่
    """
    # ถ้าไม่มี field เลย ให้ข้าม
    if not any(value is not None for value in fields.values()):
        return False
    try:
        point = influxdb_client.Point("plant_information")
        point.time(timestamp)

        for key, value in tags.items():
            if value is not None:
                point.tag(key, value)

        for key, value in fields.items():
            if value is None:
                continue
            # ถ้าค่าเป็นตัวเลข จะพยายามใช้ float (เพื่อเก็บเป็น numeric)
            # ถ้าแปลงไม่ได้ จะเขียนเป็น string (catch เพื่อรองรับ image_url เป็นต้น)
            try:
                point.field(key, float(value))
            except (ValueError, TypeError):
                point.field(key, value)

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        print(f"✅ เพิ่มข้อมูลเวลา {timestamp.strftime('%Y-%m-%d %H:%M:%S')} สำเร็จ! tags={tags} fields={fields}")
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเพิ่มข้อมูล: {e}")
        return False


def clear_bucket_data():
    """ลบข้อมูลทั้งหมดใน measurement plant_information (ระวังการใช้งาน)"""
    client = None
    try:
        client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

        try:
            buckets_api = client.buckets_api()
            buckets = buckets_api.find_buckets().buckets
            bucket_names = [b.name for b in buckets]
            print("Buckets available for this token:", bucket_names)
            if INFLUX_BUCKET not in bucket_names:
                print(f"⚠️ ชื่อ bucket '{INFLUX_BUCKET}' ไม่พบใน token นี้ — ตรวจสอบชื่อ bucket หรือสิทธิ์ token")
        except Exception as e:
            print("⚠️ ไม่สามารถดึงรายชื่อ buckets — token อาจไม่มีสิทธิ์ read/list:", e)

        delete_api = client.delete_api()
        start_time = "1970-01-01T00:00:00Z"
        stop_time = (datetime.now(timezone.utc) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        predicate = '_measurement="plant_information"'

        print(f"กำลังลบข้อมูลจาก bucket='{INFLUX_BUCKET}', measurement=plant_information, start={start_time}, stop={stop_time}")
        delete_api.delete(start=start_time, stop=stop_time, predicate=predicate, bucket=INFLUX_BUCKET, org=INFLUX_ORG)

        # ยืนยันด้วยการ query ตัวอย่าง
        query_api = client.query_api()
        flux = f'''
        from(bucket:"{INFLUX_BUCKET}")
          |> range(start: 1970-01-01T00:00:00Z, stop: {stop_time})
          |> filter(fn: (r) => r._measurement == "plant_information")
          |> limit(n:1)
        '''
        res = query_api.query(flux)
        if not res or len(res) == 0:
            print("✨ ลบข้อมูลสำเร็จ (ไม่พบข้อมูลตัวอย่างใน measurement 'plant_information').")
        else:
            print("⚠️ หลังลบยังพบข้อมูลตัวอย่าง — อาจเป็นเพราะสิทธิ์ไม่พอ หรือ predicate ไม่ตรง.")
            for table in res:
                for record in table.records:
                    print("ตัวอย่าง record ที่ยังพบ:", record.values)
                    break
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการลบข้อมูล: {type(e).__name__}: {e}")
    finally:
        if client:
            client.close()


def main():
    client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    print("เริ่มต้นโปรแกรม...")

    model_specs = [
        {
            'tags': {'customer': '富山環境整備', 'province': '富山県', 'model': 'GB-200T-56000SB特型'},
            'fields': {'image_url': '/static/plant/富山環境整備環境整備.jpg'}
        },
        {
            'tags': {'customer': 'ジェムカ', 'province': '山口県', 'model': 'GB-250T-58000特型'},
            'fields': {'image_url': '/static/plant/ジェムカ.jpg'}
        },
        {
            'tags': {'customer': 'ループ', 'province': '青森県', 'model': 'GB-150T-35000特型'},
            'fields': {'image_url': '/static/plant/ループ.jpg'}
        },
        {
            'tags': {'customer': 'ニセコ', 'province': '北海道', 'model': 'GB-30W-5500特型'},
            'fields': {'image_url': '/static/plant/ニセコ.jpg'}
        },
        {
            'tags': {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型'},
            'fields': {'image_url': '/static/plant/GB-30W-6000特型.png'}
        },
        {
            'tags': {'customer': '光陽建設', 'province': '広島県', 'model': 'GB-125T-32000特型'},
            'fields': {'image_url': '/static/plant/GB-125T-32000特型.png'}
        },
        {
            'tags': {'customer': '鈴木工業', 'province': '宮城県', 'model': 'GB-30W-12000特型'},
            'fields': {'image_url': '/static/plant/鈴木工業.jpg'}
        },
        {
            'tags': {'customer': '直富商事', 'province': '長野県', 'model': 'GB-100T-17000特型'},
            'fields': {'image_url': '/static/plant/直富商事.jpg'}
        },
        {
            'tags': {'customer': '東北黒沢建設殿', 'province': '宮城県', 'model': 'GB-100W-23000特型'},
            'fields': {'image_url': '/static/plant/東北黒沢建設殿.jpg'}
        },
        {
            'tags': {'customer': '新潟環境開発殿', 'province': '新潟県', 'model': 'GB-110T-40000特型'},
            'fields': {'image_url': '/static/plant/新潟環境開発殿.jpg'}
        },
        {
            'tags': {'customer': 'マルエス産業殿', 'province': '岐阜県', 'model': 'GB-200T-30000特型'},
            'fields': {'image_url': '/static/plant/マルエス産業殿.jpg'}
        },
        {
            'tags': {'customer': '高岡市衛生公社殿', 'province': '富山県', 'model': 'GB-60T-12000PB特型'},
            'fields': {'image_url': '/static/plant/高岡市衛生公社殿.png'}
        },
        {
            'tags': {'customer': '東名興產殿', 'province': '静岡県', 'model': 'GB-75W-27000特型'},
            'fields': {'image_url': '/static/plant/東名興產殿.png'}
        },
        {
            'tags': {'customer': 'アンビエンテ丸大殿', 'province': '北海道', 'model': 'GB-30T-11000特型'},
            'fields': {'image_url': '/static/plant/アンビエンテ丸大殿.jpg'}
        },
        {
            'tags': {'customer': 'GEF株式会社殿', 'province': '福島県', 'model': 'GB-125T-28000特型'},
            'fields': {'image_url': '/static/plant/GEF.png'}
        },
        {
            'tags': {'customer': '横浜環境保全殿', 'province': '神奈川県', 'model': 'GB-100T-24000特型'},
            'fields': {'image_url': '/static/plant/横浜環境保全殿.jpg'}
        },
        {
            'tags': {'customer': 'ミヤテック', 'province': '三重県', 'model': 'GB-100W-18000特型'},
            'fields': {'image_url': '/static/plant/ミヤテック.jpg'}
        },
    ]

    # ถ้าต้องการส่งข้อมูล sensor จริง ให้ยกคอมเมนต์ raw_data_points ด้านล่าง
    raw_data_points = [
        # ตัวอย่างเดียว — เอาไปขยายตามต้องการ
        # {'customer': '武京商会', 'province': '群馬県', 'model': 'GB-30W-6000特型',
        #  'sensor_name': '乾溜ガス化炉A', 'field_name': '温度', 'unit': '℃', 'value': 99, 'status': None},
    ]

    grouped_data = defaultdict(lambda: {"tags": {}, "fields": {}})
    for dp in raw_data_points:
        key = (dp['customer'], dp['model'], dp['sensor_name'])
        grouped_data[key]['tags'] = {
            "customer": dp['customer'], "province": dp['province'],
            "model": dp['model'], "sensor_name": dp['sensor_name']
        }
        field_name = dp['field_name']
        if dp.get('value') is not None:
            field_key = f"{field_name}_{dp['unit']}" if dp.get('unit') else field_name
            grouped_data[key]['fields'][field_key] = dp['value']
        if dp.get('status') is not None:
            grouped_data[key]['fields'][field_name] = dp['status']

    master_timestamp = datetime.now(timezone.utc)
    print(f"\n--- กำลังส่งข้อมูลด้วยเวลา: {master_timestamp.strftime('%Y-%m-%d %H:%M:%S')} ---")

    for spec in model_specs:
        write_data(write_api, tags=spec['tags'], fields=spec['fields'], timestamp=master_timestamp)

    for group in grouped_data.values():
        write_data(write_api, tags=group['tags'], fields=group['fields'], timestamp=master_timestamp)

    # clear_bucket_data()

    client.close()
    print("\nจบการทำงาน")


if __name__ == "__main__":
    main()
