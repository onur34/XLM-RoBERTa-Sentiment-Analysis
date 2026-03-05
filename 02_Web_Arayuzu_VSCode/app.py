""" 
from flask import Flask, jsonify

# Flask uygulamamızı başlatıyoruz
app = Flask(__name__)

# JSON çıktılarında Türkçe karakterlerin bozulmasını engeller
app.json.ensure_ascii = False

# Sistemin çalıştığını kontrol etmek için basit bir ana sayfa rotası
@app.route('/', methods=['GET'])
def ana_sayfa():
    return jsonify({
        "mesaj": "Duygu Analizi Backend Sistemi Başarıyla Çalışıyor!",
        "durum": "Aktif"
    })

# Sunucuyu ayağa kaldıran komut
if __name__ == '__main__':
    # debug=True sayesinde kodda değişiklik yapınca sunucu otomatik yenilenir
    app.run(debug=True, port=5000) 

"""




from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
app.json.ensure_ascii = False 


# 1. YAPAY ZEKA MODELİ ENTEGRASYONU

MODEL_YOLU = "onuru/tez-duygu-modelim"

print("Yapay Zeka Modeli ve Tercüman yükleniyor, lütfen bekleyin... ⏳")
tokenizer = AutoTokenizer.from_pretrained(MODEL_YOLU)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_YOLU)
print("Model başarıyla yüklendi! ✅")


# 2. NEDEN (SEBEP) ÇIKARMA ALGORİTMASI
import re
from collections import Counter

def sebepleri_bul(metin_listesi):
    toplam_yorum = len(metin_listesi)
    if not metin_listesi or toplam_yorum < 2:
        return "Yeterli veri bulunamadı."
        
    try:
        temiz = [str(m).lower() for m in metin_listesi]

        # ULTRA-KAPSAYICI DEVAZA MİKRO-KÖK SÖZLÜĞÜ (30 Kategori)
        temalar = {
            # 1. DİJİTAL MEDYA VE İÇERİK
            "Görüntü ve Çözünürlük": ["görüntü", "çözünürlük", "piksel", "kamera", "144p", "karanlık", "bulanık", "odak", "quality", "resolution", "pixel", "camera", "blur", "focus"],
            "Ses ve İşitsel Kalite": ["ses", "müzik", "cızırtı", "gürültü", "mikrofon", "akustik", "audio", "sound", "noise", "mic", "acoustic"],
            "Akıcı Anlatım ve Diksiyon": ["akıcı", "diksiyon", "hitabet", "anlatım", "ses tonu", "takılmadan", "fluent", "diction", "narration", "speech", "voice"],
            "İçerik ve Bilgi Verimliliği": ["öğretici", "faydalı", "zaman kaybı", "sıkıcı", "bilgi", "boş muhabbet", "informative", "useless", "waste of time", "boring"],
            "Reklam ve Sponsorluk": ["reklam", "sponsor", "araya giren", "premium", "adblock", "ads", "commercial", "sponsor"],
            
            # 2. TEKNOLOJİ, YAZILIM VE DONANIM
            "Yazılım ve Optimizasyon": ["yazılım", "güncelleme", "kası", "donu", "çök", "hız", "arayüz", "uygulama", "hata", "sürüm", "software", "update", "lag", "crash", "app", "bug", "version"],
            "İnternet ve Sunucu Bağlantısı": ["internet", "ping", "bağlantı", "sunucu", "koptu", "çekmiyor", "wifi", "fiber", "connection", "server", "disconnect", "network"],
            "Batarya ve Güç Tüketimi": ["şarj", "batarya", "pil", "ısın", "kapan", "adaptör", "battery", "charge", "heat", "power"],
            "Donanım ve Malzeme Kalitesi": ["ekran", "kasa", "işlemci", "hafıza", "anakart", "klavye", "tuş", "screen", "hardware", "processor", "memory", "keyboard"],
            
            # 3. E-TİCARET, KARGO VE ÜRÜN
            "Kargo ve Teslimat Süreci": ["kargo", "teslimat", "kurye", "paket", "gecik", "hızlı", "şube", "getirdi", "dağıtım", "delivery", "courier", "shipping", "package", "late"],
            "Ürün Sağlamlığı (Kırık/Hasar)": ["kırık", "hasar", "ezik", "çizik", "bozuk", "çalışmıyor", "defolu", "sağlam", "broken", "damaged", "defective", "scratch"],
            "İade ve Değişim Süreci": ["iade", "değişim", "kabul etmedi", "geri gönder", "garanti", "return", "exchange", "refund", "warranty"],
            "Orijinallik ve Güvenilirlik": ["orijinal", "sahte", "çakma", "replika", "güvenilir", "dolandır", "fake", "original", "scam", "replica"],
            
            # 4. GİYİM, MODA VE KOZMETİK
            "Kumaş ve Materyal Kalitesi": ["kumaş", "pamuk", "naylon", "sökük", "yırtık", "tüylen", "soldu", "çekti", "fabric", "cotton", "torn", "fade", "material"],
            "Beden ve Kalıp Uyumu": ["beden", "kalıp", "dar", "bol", "kısa", "uzun", "ölçü", "üzerime", "size", "fit", "tight", "loose"],
            "Kozmetik ve Cilt Etkisi": ["alerji", "sivilce", "koku", "nemlendir", "kuruttu", "cilt", "parfüm", "allergy", "acne", "smell", "skin", "perfume"],

            # 5. MÜŞTERİ İLİŞKİLERİ VE EKONOMİ
            "Müşteri Hizmetleri İlgisi": ["müşteri", "temsilci", "destek", "yardım", "iletişim", "saygı", "kibar", "çözüm", "şikayet", "ilgisiz", "customer", "support", "service", "rude", "polite"],
            "Fiyat ve Maliyet Oranı": ["fiyat", "pahalı", "ucuz", "indirim", "para", "ücret", "kazık", "maliyet", "ederi", "price", "expensive", "cheap", "cost", "money", "discount"],
            
            # 6. GIDA, RESTORAN VE KAFE
            "Lezzet ve Tazelik": ["lezzet", "tat", "taze", "bayat", "çiğ", "yanık", "tuz", "şeker", "taste", "flavor", "fresh", "stale", "raw", "burnt", "salty"],
            "Porsiyon ve Doyuruculuk": ["porsiyon", "doyurucu", "az", "küçük", "gramaj", "avuç", "portion", "filling", "small", "size"],
            "Mekan Hijyeni ve Servis": ["hijyen", "temiz", "pis", "garson", "servis", "mekan", "tuvalet", "masa", "hygiene", "clean", "dirty", "waiter", "place", "table"],
            
            # 7. SİYASET, GÜNDEM VE TOPLUM
            "Siyaset ve Yönetim": ["seçim", "siyaset", "parti", "oy ", "hükümet", "muhalefet", "bakan", "başkan", "meclis", "politika", "politics", "election", "government", "vote"],
            "Toplumsal Gündem ve Sosyal Medya": ["linç", "gündem", "tweet", "fenomen", "adalet", "hukuk", "trend", "viral", "troll", "justice", "law", "trend"],
            
            # 8. SPOR VE OYUN DÜNYASI
            "Hakem ve Spor Yönetimi": ["hakem", "penaltı", "kart", "ofsayt", "taraftar", "derbi", "maç", "referee", "penalty", "fans", "match", "derby"],
            "Oyun İçi Denge ve Hileciler": ["hile", "bug", "haksızlık", "nerf", "buff", "eşleştirme", "oyuncu", "cheat", "hacker", "matchmaking", "player"],
            
            # 9. FİNANS, BANKA VE KRİPTO
            "Borsa, Kripto ve Yatırım": ["hisse", "kripto", "coin", "zarar", "kar ", "borsa", "enflasyon", "yatırım", "stock", "crypto", "profit", "loss", "investment", "inflation"],
            "Banka, Kredi ve Kesintiler": ["banka", "kredi", "faiz", "kesinti", "komisyon", "kart", "limit", "hesap", "bank", "credit", "interest", "commission", "account"],
            
            # 10. EĞİTİM, SAĞLIK VE ULAŞIM
            "Eğitim, Sınav ve Öğretmen": ["hoca", "öğretmen", "ders", "sınav", "müfredat", "okul", "eğitim", "öğrenci", "teacher", "exam", "course", "education", "student"],
            "Sağlık, Hastane ve Tedavi": ["doktor", "randevu", "tahlil", "hastane", "muayene", "ilaç", "ameliyat", "doctor", "hospital", "appointment", "surgery", "medicine"],
            "Trafik, Araç ve Ulaşım": ["trafik", "kaza", "otobüs", "şoför", "rötar", "uçuş", "bilet", "metro", "yol", "traffic", "accident", "bus", "flight", "delay", "ticket", "road"]
        }

        tema_sayilari = {}
        for tema, kelimeler in temalar.items():
            sayac = 0
            for metin in temiz:
                if any(kelime in metin for kelime in kelimeler):
                    sayac += 1
            if sayac > 0:
                tema_sayilari[tema] = sayac

        if not tema_sayilari:
            return "Belirgin bir spesifik sebep kelimesi tespit edilemedi."

        # En çok geçen mikro-nedeni bul
        en_iyi_tema = max(tema_sayilari, key=tema_sayilari.get)
        kac_gecti = tema_sayilari[en_iyi_tema]
        
        # Temiz yüzde hesabı
        yuzde = (kac_gecti / toplam_yorum) * 100
        if yuzde > 100: yuzde = 100

        return f"Yorumların %{yuzde:.0f}'sinde ana faktör olarak '{en_iyi_tema}' öne çıkıyor."

    except Exception as e:
        return "Sebep analizi yapılamadı."


# 3. WEB SAYFASI VE API ROTALARI

@app.route('/', methods=['GET'])
def ana_sayfa():
    return render_template('index.html')

# TEKİL TAHMİN MOTORU 
@app.route('/tahmin', methods=['POST'])
def tahmin_yap():
    try:
        veri = request.get_json()
        metin = veri.get('metin', '')
        if not metin: return jsonify({'hata': 'Metin boş.'}), 400

        girdiler = tokenizer(metin, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad(): ciktilar = model(**girdiler)
        
        olasiliklar = torch.nn.functional.softmax(ciktilar.logits, dim=-1)
        tahmin = torch.argmax(olasiliklar, dim=-1).item()
        guven = olasiliklar[0][tahmin].item() * 100

        return jsonify({'metin': metin, 'duygu': "Olumlu" if tahmin == 1 else "Olumsuz", 'guven_skoru': f"%{guven:.2f}"})
    except Exception as e:
        return jsonify({'hata': str(e)}), 500

# TOPLU TAHMİN VE SEBEP MOTORU
@app.route('/toplu-analiz', methods=['POST'])
def toplu_analiz():
    try:
        if 'dosya' not in request.files:
            return jsonify({'hata': 'Lütfen bir CSV dosyası yükleyin.'}), 400
            
        dosya = request.files['dosya']
        df = pd.read_csv(dosya)
        
        # Dosyada 'metin', 'yorum' veya 'comment' başlıklı sütunu arıyoruz
        metin_sutunu = next((kolon for kolon in df.columns if kolon.lower() in ['metin', 'yorum', 'comment', 'review']), None)
                
        if not metin_sutunu:
            return jsonify({'hata': 'Excel/CSV dosyasında "metin", "yorum" veya "comment" isimli bir sütun bulunamadı!'}), 400

        # Boş satırları sil ve en fazla 200 yorumu al (Sistem hızlı çalışsın diye)
        metinler = df[metin_sutunu].dropna().astype(str).tolist()[:200] 
        
        olumlu_metinler, olumsuz_metinler = [], []
        
        for metin in metinler:
            girdiler = tokenizer(metin, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                sinif = torch.argmax(model(**girdiler).logits, dim=-1).item()
            if sinif == 1: olumlu_metinler.append(metin)
            else: olumsuz_metinler.append(metin)
                
        toplam = len(metinler)
        olumlu_yuzde = (len(olumlu_metinler) / toplam) * 100 if toplam > 0 else 0
        olumsuz_yuzde = (len(olumsuz_metinler) / toplam) * 100 if toplam > 0 else 0
        
        return jsonify({
            'toplam_yorum': toplam,
            'olumlu_orani': f"%{olumlu_yuzde:.1f}",
            'olumsuz_orani': f"%{olumsuz_yuzde:.1f}",
            'olumlu_sebepler': sebepleri_bul(olumlu_metinler),
            'olumsuz_sebepler': sebepleri_bul(olumsuz_metinler)
        })

    except Exception as e:
        return jsonify({'hata': f"Dosya okuma hatası: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)