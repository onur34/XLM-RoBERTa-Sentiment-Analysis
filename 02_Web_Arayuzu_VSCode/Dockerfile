# 1. İçinde Python yüklü olan hafif bir sunucu tabanı seçiyoruz
FROM python:3.9-slim

# 2. Sunucunun içinde kendimize bir çalışma klasörü oluşturuyoruz
WORKDIR /app

# 3. İhtiyaç listemizi (requirements.txt) sunucuya kopyalıyoruz
COPY requirements.txt .

# 4. Listeyi okuyup gerekli kütüphaneleri sunucuya kurmasını söylüyoruz
RUN pip install --no-cache-dir -r requirements.txt

# 5. Bizim model klasörümüzü, app.py ve arayüz dosyalarımızı sunucuya kopyalıyoruz
COPY . .

# 6. Hugging Face sunucularının dış dünya ile iletişim kurduğu 7860 portunu açıyoruz
EXPOSE 7860

# 7. Sistemi test motoruyla değil, güçlü Gunicorn motoruyla başlatıyoruz (Zaman aşımı süresini 120 sn yapıyoruz ki model yüklenirken sunucu kapanmasın)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]