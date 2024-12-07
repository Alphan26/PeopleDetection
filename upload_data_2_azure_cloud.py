import os
import random
from azure.storage.blob import BlobServiceClient

# Azure Storage account bilgileri
account_name = "alpobjdetection"
container_name = "realtimeobjtracking"


# Klasör yolları (birden fazla klasör olabilir)
veriler_path = "/home/alphan/İndirilenler/MOT20"
etiketler_path = "/home/alphan/İndirilenler/MOT20Labels"

# BlobServiceClient nesnesi oluşturma
blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)

# Her klasördeki dosyaları %20 oranında yükleme fonksiyonu
def upload_files_to_blob(folder_path, percentage=20):
    container_client = blob_service_client.get_container_client(container_name)
    
    # Klasördeki tüm alt klasörleri ve dosyaları listele
    for root, dirs, files in os.walk(folder_path):
        all_files = [os.path.join(root, file) for file in files]
        
        # Dosyaların %20'sini seç
        files_to_upload = random.sample(all_files, int(len(all_files) * percentage / 100)) if all_files else []
        
        # Seçilen dosyaları yükle
        for file_path in files_to_upload:
            blob_name = os.path.relpath(file_path, folder_path)
            blob_client = container_client.get_blob_client(blob_name)
            
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                print(f"Yüklenen: {file_path} -> {blob_name}")

# Veriler ve Etiketler klasörlerinin her birinden %10'sini yükle
upload_files_to_blob(veriler_path, percentage=10)
upload_files_to_blob(etiketler_path, percentage=10)
