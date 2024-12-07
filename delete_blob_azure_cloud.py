from azure.storage.blob import BlobServiceClient

# Azure Storage account bilgileri
account_name = "alpobjdetection"
container_name = "realtimeobjtracking"


# Tüm blob'ları silme fonksiyonu
def delete_all_blobs():
    container_client = blob_service_client.get_container_client(container_name)
    
    # Container'daki tüm blob'ları listele
    blobs = container_client.list_blobs()
    
    # Her bir blob'ı sil
    for blob in blobs:
        blob_client = container_client.get_blob_client(blob.name)
        blob_client.delete_blob()
        print(f"Silinen blob: {blob.name}")

# Container'daki tüm blob'ları sil
delete_all_blobs()
